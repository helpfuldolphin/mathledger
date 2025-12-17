# Curriculum Phase X Invariants: Allowed and Forbidden Transitions

---

> **SYSTEM LAW DOCUMENT — CURRICULUM EVOLUTION CONSTRAINTS**
>
> This document specifies the invariants governing curriculum slice transitions
> during Phase X P3 and P4 shadow operations. All curriculum mutations must
> comply with these rules or be rejected.
>
> **SHADOW MODE: These invariants are enforced regardless of observation mode.**

---

**Status**: Specification
**Version**: 1.0.0
**Date**: 2025-12-10
**Scope**: Phase X P3/P4 Curriculum Governance

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Curriculum Transition Model](#2-curriculum-transition-model)
3. [Phase X P3 Invariants](#3-phase-x-p3-invariants)
4. [Phase X P4 Invariants](#4-phase-x-p4-invariants)
5. [Allowed Transitions](#5-allowed-transitions)
6. [Forbidden Transitions](#6-forbidden-transitions)
7. [Monotonicity Constraints](#7-monotonicity-constraints)
8. [Gate Threshold Evolution Rules](#8-gate-threshold-evolution-rules)
9. [Drift Classification Matrix](#9-drift-classification-matrix)
10. [Enforcement Mechanisms](#10-enforcement-mechanisms)
11. [Schema Definitions](#11-schema-definitions)
12. [Phase X P5 Considerations: Real Telemetry Stress](#12-phase-x-p5-considerations-real-telemetry-stress)
13. [Curriculum Coherence Summary](#13-curriculum-coherence-summary)
14. [CTRPK Integration Plan: Global Health, Evidence, and Council](#14-ctrpk-integration-plan-global-health-evidence-and-council)
15. [Smoke-Test Readiness Checklist](#15-smoke-test-readiness-checklist)

---

## 1. Executive Summary

Curriculum slices define the bounded complexity regions within which the MathLedger derivation engine operates. Each slice specifies:

- **Parameters**: `atoms`, `depth_max`, `breadth_max`, `total_max`
- **Gates**: Coverage, abstention, velocity, caps thresholds

During Phase X (P3/P4), curriculum transitions are constrained by shadow mode invariants. This document codifies:

1. **What transitions are allowed** — monotonic advancement, parameter tightening
2. **What transitions are forbidden** — regression, gate relaxation, skip-ahead
3. **How drift is classified** — NONE, PARAMETRIC, SEMANTIC severity levels
4. **How violations are handled** — BLOCK, WARN, LOG status responses

### Core Principle

**Curriculum evolution must be monotonically non-regressing.** A system that has proven capability at complexity level N must not retreat to level N-1 without explicit governance override.

---

## 2. Curriculum Transition Model

### 2.1 State Space

A curriculum state `C` is defined as:

```
C = (S, A, P, G)

where:
  S = slice_name (string identifier)
  A = active_index (0-indexed position in slice ladder)
  P = params {atoms, depth_max, breadth_max, total_max}
  G = gates {coverage, abstention, velocity, caps}
```

### 2.2 Transition Function

A curriculum transition `T: C → C'` transforms the current state to a new state:

```
T(C) = C' iff:
  1. Monotonicity(P, P') holds
  2. GateEvolution(G, G') is valid
  3. IndexAdvancement(A, A') is legal
  4. SliceSequence(S, S') is contiguous
```

### 2.3 Transition Types

| Type | Symbol | Description |
|------|--------|-------------|
| ADVANCE | `→+` | Move to next slice in ladder |
| HOLD | `→=` | Remain at current slice |
| RETUNE | `→~` | Modify gate thresholds within slice |
| REGRESS | `→-` | **FORBIDDEN** — Move to previous slice |

---

## 3. Phase X P3 Invariants

Phase X P3 operates in **synthetic shadow mode** where curriculum state is observed but not mutated by experiment execution.

### 3.1 P3 Curriculum Invariants Table

| ID | Invariant | Enforcement | Violation Severity |
|----|-----------|-------------|-------------------|
| CUR-P3-01 | Slice ladder is read-only during shadow runs | Code assertion | CRITICAL |
| CUR-P3-02 | Gate thresholds are frozen at run start | Config snapshot | CRITICAL |
| CUR-P3-03 | No `completed_at` timestamps written during shadow | Write guard | CRITICAL |
| CUR-P3-04 | Synthetic metrics do not trigger ratchet evaluation | Logic guard | CRITICAL |
| CUR-P3-05 | Curriculum fingerprint is immutable per run | Hash validation | WARNING |

### 3.2 P3 Shadow Isolation Contract

```python
class P3CurriculumContract:
    """
    P3 curriculum isolation invariants.

    SHADOW MODE: Curriculum is OBSERVED, never MUTATED.
    """

    # INV: CUR-P3-01 — No slice ladder modifications
    def assert_ladder_frozen(self, before: List[str], after: List[str]) -> None:
        if before != after:
            raise CurriculumViolation("CUR-P3-01: Slice ladder modified during P3 shadow run")

    # INV: CUR-P3-02 — Gate thresholds unchanged
    def assert_gates_frozen(self, before: Dict, after: Dict) -> None:
        if before != after:
            raise CurriculumViolation("CUR-P3-02: Gate thresholds modified during P3 shadow run")

    # INV: CUR-P3-03 — No completion timestamps
    def assert_no_completion_writes(self, slice_obj: CurriculumSlice) -> None:
        # completed_at must remain None or unchanged from run start
        pass

    # INV: CUR-P3-04 — No ratchet evaluation
    def assert_no_ratchet_calls(self) -> None:
        # should_ratchet() must not be called with shadow metrics
        pass
```

---

## 4. Phase X P4 Invariants

Phase X P4 operates in **real runner shadow mode** where curriculum state is observed via read-only adapter but transitions are still forbidden.

### 4.1 P4 Curriculum Invariants Table

| ID | Invariant | Enforcement | Violation Severity |
|----|-----------|-------------|-------------------|
| CUR-P4-01 | All P3 invariants remain in force | Inheritance | CRITICAL |
| CUR-P4-02 | Real runner curriculum reads are non-mutating | Adapter contract | CRITICAL |
| CUR-P4-03 | Twin curriculum state divergence is logged only | No remediation | CRITICAL |
| CUR-P4-04 | No `activate_next_slice()` calls in P4 | Function guard | CRITICAL |
| CUR-P4-05 | Curriculum drift events recorded to audit log | JSONL schema | WARNING |

### 4.2 P4 Read-Only Adapter Contract

```python
class P4CurriculumAdapterContract:
    """
    P4 curriculum adapter invariants.

    SHADOW MODE: Read-only observation of real curriculum state.
    """

    # INV: CUR-P4-02 — Non-mutating reads
    ALLOWED_METHODS = frozenset([
        "load",
        "active_slice",
        "active_name",
        "next_slice",
        "to_public_dict",
    ])

    FORBIDDEN_METHODS = frozenset([
        "activate_next_slice",
        "should_ratchet",  # May trigger side effects
    ])

    def validate_method_call(self, method_name: str) -> None:
        if method_name in self.FORBIDDEN_METHODS:
            raise CurriculumViolation(f"CUR-P4-02: Forbidden method '{method_name}' in P4 shadow mode")
```

---

## 5. Allowed Transitions

### 5.1 Transition Matrix

| From State | To State | Allowed | Condition |
|------------|----------|---------|-----------|
| Slice N | Slice N | YES | HOLD — no change |
| Slice N | Slice N+1 | YES | ADVANCE — all gates passed |
| Slice N params | Slice N params (stricter) | YES | RETUNE — parameter tightening |
| Slice N gates | Slice N gates (stricter) | YES | RETUNE — threshold tightening |

### 5.2 Allowed Parameter Evolutions

| Parameter | Allowed Direction | Rationale |
|-----------|-------------------|-----------|
| `atoms` | `≥` (non-decreasing) | Complexity expansion only |
| `depth_max` | `≥` (non-decreasing) | Derivation depth expansion only |
| `breadth_max` | `≥` (non-decreasing) | Search breadth expansion only |
| `total_max` | `≥` (non-decreasing) | Capacity expansion only |

### 5.3 Allowed Gate Evolutions

| Gate | Threshold | Allowed Direction | Rationale |
|------|-----------|-------------------|-----------|
| Coverage | `ci_lower_min` | `≥` (non-decreasing) | Stricter coverage requirement |
| Coverage | `sample_min` | `≥` (non-decreasing) | Larger sample requirement |
| Abstention | `max_rate_pct` | `≤` (non-increasing) | Stricter abstention limit |
| Abstention | `max_mass` | `≤` (non-increasing) | Stricter mass limit |
| Velocity | `min_pph` | `≥` (non-decreasing) | Higher velocity floor |
| Velocity | `stability_cv_max` | `≤` (non-increasing) | Stricter stability |
| Caps | `min_attempt_mass` | `≥` (non-decreasing) | Larger minimum mass |
| Caps | `min_runtime_minutes` | `≥` (non-decreasing) | Longer minimum runtime |
| Caps | `backlog_max` | `≤` (non-increasing) | Stricter backlog limit |

---

## 6. Forbidden Transitions

### 6.1 Forbidden Transition Matrix

| From State | To State | Forbidden | Violation Type |
|------------|----------|-----------|----------------|
| Slice N | Slice N-1 | **FORBIDDEN** | REGRESS |
| Slice N | Slice N+2 | **FORBIDDEN** | SKIP |
| Slice N params | Slice N params (relaxed) | **FORBIDDEN** | PARAM_REGRESS |
| Slice N gates | Slice N gates (relaxed) | **FORBIDDEN** | GATE_REGRESS |
| Any | Delete slice | **FORBIDDEN** | LADDER_MUTATION |
| Any | Insert slice (non-terminal) | **FORBIDDEN** | LADDER_MUTATION |

### 6.2 Forbidden Parameter Evolutions

| Parameter | Forbidden Direction | Violation Code |
|-----------|---------------------|----------------|
| `atoms` | `<` (decrease) | PARAM_REGRESS_ATOMS |
| `depth_max` | `<` (decrease) | PARAM_REGRESS_DEPTH |
| `breadth_max` | `<` (decrease) | PARAM_REGRESS_BREADTH |
| `total_max` | `<` (decrease) | PARAM_REGRESS_TOTAL |

### 6.3 Forbidden Gate Evolutions

| Gate | Threshold | Forbidden Direction | Violation Code |
|------|-----------|---------------------|----------------|
| Coverage | `ci_lower_min` | `<` (decrease) | GATE_REGRESS_COV_CI |
| Coverage | `sample_min` | `<` (decrease) | GATE_REGRESS_COV_SAMPLE |
| Abstention | `max_rate_pct` | `>` (increase) | GATE_REGRESS_ABS_RATE |
| Abstention | `max_mass` | `>` (increase) | GATE_REGRESS_ABS_MASS |
| Velocity | `min_pph` | `<` (decrease) | GATE_REGRESS_VEL_PPH |
| Velocity | `stability_cv_max` | `>` (increase) | GATE_REGRESS_VEL_CV |
| Caps | `min_attempt_mass` | `<` (decrease) | GATE_REGRESS_CAP_MASS |
| Caps | `min_runtime_minutes` | `<` (decrease) | GATE_REGRESS_CAP_RT |
| Caps | `backlog_max` | `>` (increase) | GATE_REGRESS_CAP_BL |

### 6.4 Special Forbidden Cases

| Case | Description | Violation Code |
|------|-------------|----------------|
| Active slice deletion | Removing the currently active slice | ACTIVE_SLICE_DELETE |
| Completed slice modification | Changing params/gates of completed slice | SEALED_SLICE_MODIFY |
| Attestation removal | Deleting attestation records | ATTESTATION_TAMPER |
| Version downgrade | Changing curriculum version < current | VERSION_REGRESS |

---

## 7. Monotonicity Constraints

### 7.1 Formal Definition

For a curriculum system with monotonic axes `M = {axis_1, ..., axis_k}`, a transition `C → C'` satisfies monotonicity iff:

```
∀ axis ∈ M: C'.params[axis] ≥ C.params[axis]
```

### 7.2 Monotonic Axes Configuration

The standard monotonic axes for MathLedger curriculum:

```yaml
invariants:
  monotonic_axes:
    - atoms
    - depth_max
```

### 7.3 Monotonicity Violation Detection

```python
def check_monotonicity(
    before_slice: CurriculumSlice,
    after_slice: CurriculumSlice,
    monotonic_axes: Tuple[str, ...]
) -> List[MonotonicityViolation]:
    """
    Check for monotonicity violations between slice transitions.

    Returns:
        List of violations (empty if compliant)
    """
    violations = []
    for axis in monotonic_axes:
        before_val = before_slice.params.get(axis)
        after_val = after_slice.params.get(axis)

        if before_val is None or after_val is None:
            violations.append(MonotonicityViolation(
                axis=axis,
                type="MISSING_AXIS",
                before=before_val,
                after=after_val,
            ))
        elif after_val < before_val:
            violations.append(MonotonicityViolation(
                axis=axis,
                type="REGRESSION",
                before=before_val,
                after=after_val,
                delta=after_val - before_val,
            ))

    return violations
```

---

## 8. Gate Threshold Evolution Rules

### 8.1 Constraint Direction Matrix

| Gate Category | Threshold | Constraint | Allowed Δ |
|---------------|-----------|------------|-----------|
| Coverage | `ci_lower_min` | INCREASING | `≥ 0` |
| Coverage | `sample_min` | INCREASING | `≥ 0` |
| Coverage | `require_attestation` | BOOLEAN_TRUE | `False → True` only |
| Abstention | `max_rate_pct` | DECREASING | `≤ 0` |
| Abstention | `max_mass` | DECREASING | `≤ 0` |
| Velocity | `min_pph` | INCREASING | `≥ 0` |
| Velocity | `stability_cv_max` | DECREASING | `≤ 0` |
| Velocity | `window_minutes` | ANY | No constraint |
| Caps | `min_attempt_mass` | INCREASING | `≥ 0` |
| Caps | `min_runtime_minutes` | INCREASING | `≥ 0` |
| Caps | `backlog_max` | DECREASING | `≤ 0` |

### 8.2 Evolution Validation Function

```python
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

Constraint = Literal["increasing", "decreasing", "boolean_true", "any"]

@dataclass(frozen=True)
class GateEvolutionRule:
    path: Tuple[str, ...]
    constraint: Constraint

GATE_EVOLUTION_RULES = [
    GateEvolutionRule(("coverage", "ci_lower_min"), "increasing"),
    GateEvolutionRule(("coverage", "sample_min"), "increasing"),
    GateEvolutionRule(("coverage", "require_attestation"), "boolean_true"),
    GateEvolutionRule(("abstention", "max_rate_pct"), "decreasing"),
    GateEvolutionRule(("abstention", "max_mass"), "decreasing"),
    GateEvolutionRule(("velocity", "min_pph"), "increasing"),
    GateEvolutionRule(("velocity", "stability_cv_max"), "decreasing"),
    GateEvolutionRule(("velocity", "window_minutes"), "any"),
    GateEvolutionRule(("caps", "min_attempt_mass"), "increasing"),
    GateEvolutionRule(("caps", "min_runtime_minutes"), "increasing"),
    GateEvolutionRule(("caps", "backlog_max"), "decreasing"),
]

def validate_gate_evolution(
    before_gates: Dict,
    after_gates: Dict,
) -> Tuple[bool, List[str]]:
    """
    Validate that gate evolution follows constraint rules.

    Returns:
        (is_valid, list of violation descriptions)
    """
    violations = []

    for rule in GATE_EVOLUTION_RULES:
        before_val = _get_nested(before_gates, rule.path)
        after_val = _get_nested(after_gates, rule.path)

        if before_val is None or after_val is None:
            continue

        if rule.constraint == "increasing":
            if after_val < before_val:
                violations.append(
                    f"{'.'.join(rule.path)}: {before_val} → {after_val} (must be increasing)"
                )
        elif rule.constraint == "decreasing":
            if after_val > before_val:
                violations.append(
                    f"{'.'.join(rule.path)}: {before_val} → {after_val} (must be decreasing)"
                )
        elif rule.constraint == "boolean_true":
            if before_val and not after_val:
                violations.append(
                    f"{'.'.join(rule.path)}: True → False (cannot disable)"
                )

    return (len(violations) == 0, violations)
```

---

## 9. Drift Classification Matrix

### 9.1 Severity Levels

| Severity | Code | Description | Response |
|----------|------|-------------|----------|
| NONE | `0` | No drift detected | OK — proceed |
| PARAMETRIC | `1` | Parameter changed within allowed direction | WARN — log |
| SEMANTIC | `2` | Constraint direction violated | BLOCK — reject |

### 9.2 Classification Rules

```python
def classify_drift(
    path: str,
    baseline: Any,
    current: Any,
    constraint: Constraint,
) -> Severity:
    """
    Classify the severity of a curriculum drift event.
    """
    if baseline == current:
        return Severity.NONE

    # Parameter changed — at minimum PARAMETRIC
    if constraint == "any":
        return Severity.PARAMETRIC

    if constraint == "boolean_true":
        if baseline and not current:
            return Severity.SEMANTIC  # Disabled required flag
        return Severity.PARAMETRIC

    if constraint == "increasing":
        if current < baseline:
            return Severity.SEMANTIC  # Regression
        return Severity.PARAMETRIC

    if constraint == "decreasing":
        if current > baseline:
            return Severity.SEMANTIC  # Relaxation
        return Severity.PARAMETRIC

    return Severity.PARAMETRIC
```

### 9.3 Status Mapping

| Severity | Status | Action |
|----------|--------|--------|
| NONE | OK | Continue execution |
| PARAMETRIC | WARN | Log warning, continue |
| SEMANTIC | BLOCK | Log error, halt transition |

---

## 10. Enforcement Mechanisms

### 10.1 Pre-Transition Validation

Before any curriculum transition:

```python
def validate_transition(
    current: CurriculumSystem,
    proposed: CurriculumSystem,
    phase: Literal["P3", "P4"],
) -> TransitionValidation:
    """
    Validate a proposed curriculum transition against Phase X invariants.
    """
    violations = []

    # Check phase-specific invariants
    if phase == "P3":
        # CUR-P3-01: Ladder frozen
        if len(current.slices) != len(proposed.slices):
            violations.append(Violation("CUR-P3-01", "Slice count changed"))

        # CUR-P3-02: Gates frozen
        for i, (cur, prop) in enumerate(zip(current.slices, proposed.slices)):
            if _gates_differ(cur.gates, prop.gates):
                violations.append(Violation("CUR-P3-02", f"Gates changed for slice {i}"))

    elif phase == "P4":
        # CUR-P4-01: All P3 invariants
        p3_result = validate_transition(current, proposed, "P3")
        violations.extend(p3_result.violations)

    # Check monotonicity (both phases)
    mono_violations = check_monotonicity_across_ladder(proposed.slices, proposed.monotonic_axes)
    violations.extend(mono_violations)

    # Check gate evolution rules (both phases)
    gate_violations = check_gate_evolution_across_ladder(current.slices, proposed.slices)
    violations.extend(gate_violations)

    return TransitionValidation(
        valid=len(violations) == 0,
        violations=violations,
        max_severity=_max_severity(violations),
    )
```

### 10.2 Runtime Guards

```python
class CurriculumRuntimeGuard:
    """
    Runtime enforcement of curriculum invariants.
    """

    def __init__(self, phase: Literal["P3", "P4"]):
        self.phase = phase
        self._snapshot: Optional[CurriculumSystem] = None

    def capture_snapshot(self, system: CurriculumSystem) -> None:
        """Capture curriculum state at run start."""
        self._snapshot = deepcopy(system)

    def verify_unchanged(self, system: CurriculumSystem) -> None:
        """Verify curriculum matches snapshot (P3/P4 shadow mode)."""
        if self._snapshot is None:
            raise CurriculumViolation("No snapshot captured")

        validation = validate_transition(self._snapshot, system, self.phase)
        if not validation.valid:
            raise CurriculumViolation(
                f"Curriculum modified during {self.phase} shadow run: "
                f"{validation.violations}"
            )
```

### 10.3 Audit Logging

All curriculum state observations and attempted transitions are logged:

```python
def log_curriculum_event(
    event_type: str,
    curriculum_fingerprint: str,
    slice_name: str,
    drift_snapshot: Dict[str, Any],
    phase: str,
) -> None:
    """
    Log curriculum event to audit trail.
    """
    event = {
        "schema": "curriculum-evolution-event/1.0.0",
        "event_type": event_type,
        "phase": phase,
        "mode": "SHADOW",
        "curriculum_fingerprint": curriculum_fingerprint,
        "slice_name": slice_name,
        "drift_status": drift_snapshot.get("status", "UNKNOWN"),
        "drift_severity": drift_snapshot.get("severity", "UNKNOWN"),
        "changed_params": drift_snapshot.get("changed_params", []),
        "timestamp": deterministic_isoformat(),
    }
    # Write to curriculum_events.jsonl
```

---

## 11. Schema Definitions

### 11.1 Curriculum Drift Timeline Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "curriculum-drift-timeline/1.0.0",
  "title": "Curriculum Drift Timeline Event",
  "description": "Records a curriculum drift observation in the timeline",
  "type": "object",
  "required": [
    "schema",
    "event_id",
    "timestamp",
    "phase",
    "mode",
    "curriculum_fingerprint",
    "slice_name",
    "drift_status",
    "drift_severity"
  ],
  "properties": {
    "schema": {
      "type": "string",
      "const": "curriculum-drift-timeline/1.0.0"
    },
    "event_id": {
      "type": "string",
      "description": "Unique event identifier (UUID)"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 timestamp of the drift observation"
    },
    "phase": {
      "type": "string",
      "enum": ["P3", "P4"],
      "description": "Phase X phase during which drift was observed"
    },
    "mode": {
      "type": "string",
      "const": "SHADOW",
      "description": "Must always be SHADOW in Phase X"
    },
    "curriculum_fingerprint": {
      "type": "string",
      "description": "SHA-256 hash of the curriculum configuration"
    },
    "slice_name": {
      "type": "string",
      "description": "Name of the active curriculum slice"
    },
    "baseline_slice_name": {
      "type": "string",
      "description": "Name of the baseline slice for comparison"
    },
    "drift_status": {
      "type": "string",
      "enum": ["OK", "WARN", "BLOCK"],
      "description": "Resulting status from drift analysis"
    },
    "drift_severity": {
      "type": "string",
      "enum": ["NONE", "PARAMETRIC", "SEMANTIC"],
      "description": "Maximum severity of detected drift"
    },
    "changed_params": {
      "type": "array",
      "items": {
        "$ref": "#/definitions/ChangedParam"
      },
      "description": "List of changed parameters"
    },
    "monotonicity_violations": {
      "type": "array",
      "items": {
        "$ref": "#/definitions/MonotonicityViolation"
      },
      "description": "List of monotonicity constraint violations"
    },
    "gate_evolution_violations": {
      "type": "array",
      "items": {
        "$ref": "#/definitions/GateEvolutionViolation"
      },
      "description": "List of gate evolution rule violations"
    },
    "action_taken": {
      "type": "string",
      "const": "LOGGED_ONLY",
      "description": "Always LOGGED_ONLY in Phase X shadow mode"
    }
  },
  "definitions": {
    "ChangedParam": {
      "type": "object",
      "required": ["path", "baseline", "current", "classification"],
      "properties": {
        "path": {
          "type": "string",
          "description": "Dot-separated path to the changed parameter"
        },
        "baseline": {
          "description": "Baseline value"
        },
        "current": {
          "description": "Current value"
        },
        "delta": {
          "type": "number",
          "description": "Numeric delta (if applicable)"
        },
        "classification": {
          "type": "string",
          "enum": ["NONE", "PARAMETRIC", "SEMANTIC"]
        },
        "constraint": {
          "type": "string",
          "enum": ["increasing", "decreasing", "boolean_true", "any"]
        }
      }
    },
    "MonotonicityViolation": {
      "type": "object",
      "required": ["axis", "type", "before", "after"],
      "properties": {
        "axis": {
          "type": "string",
          "description": "Name of the monotonic axis"
        },
        "type": {
          "type": "string",
          "enum": ["MISSING_AXIS", "REGRESSION"]
        },
        "before": {
          "type": ["number", "null"]
        },
        "after": {
          "type": ["number", "null"]
        },
        "delta": {
          "type": "number"
        }
      }
    },
    "GateEvolutionViolation": {
      "type": "object",
      "required": ["gate", "threshold", "violation_type", "before", "after"],
      "properties": {
        "gate": {
          "type": "string",
          "enum": ["coverage", "abstention", "velocity", "caps"]
        },
        "threshold": {
          "type": "string"
        },
        "violation_type": {
          "type": "string",
          "enum": ["REGRESSION", "RELAXATION", "DISABLE"]
        },
        "before": {},
        "after": {},
        "delta": {
          "type": "number"
        }
      }
    }
  }
}
```

### 11.2 Curriculum Governance Signal Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "curriculum-governance-signal/1.0.0",
  "title": "Curriculum Governance Signal",
  "description": "Emitted when curriculum governance action is required or observed",
  "type": "object",
  "required": [
    "schema",
    "signal_id",
    "timestamp",
    "phase",
    "mode",
    "signal_type",
    "curriculum_fingerprint",
    "active_slice",
    "severity",
    "status"
  ],
  "properties": {
    "schema": {
      "type": "string",
      "const": "curriculum-governance-signal/1.0.0"
    },
    "signal_id": {
      "type": "string",
      "description": "Unique signal identifier (UUID)"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time"
    },
    "phase": {
      "type": "string",
      "enum": ["P3", "P4"]
    },
    "mode": {
      "type": "string",
      "const": "SHADOW"
    },
    "signal_type": {
      "type": "string",
      "enum": [
        "DRIFT_DETECTED",
        "TRANSITION_REQUESTED",
        "TRANSITION_VALIDATED",
        "TRANSITION_BLOCKED",
        "INVARIANT_VIOLATION",
        "SNAPSHOT_CAPTURED",
        "SNAPSHOT_VERIFIED"
      ],
      "description": "Type of governance signal"
    },
    "curriculum_fingerprint": {
      "type": "string"
    },
    "active_slice": {
      "type": "string"
    },
    "target_slice": {
      "type": "string",
      "description": "Target slice for transition signals"
    },
    "severity": {
      "type": "string",
      "enum": ["NONE", "PARAMETRIC", "SEMANTIC"]
    },
    "status": {
      "type": "string",
      "enum": ["OK", "WARN", "BLOCK"]
    },
    "violations": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["code", "message"],
        "properties": {
          "code": {
            "type": "string"
          },
          "message": {
            "type": "string"
          },
          "details": {
            "type": "object"
          }
        }
      }
    },
    "governance_action": {
      "type": "string",
      "enum": ["LOGGED_ONLY", "WOULD_BLOCK", "WOULD_WARN"],
      "description": "Action taken (always observational in Phase X)"
    },
    "hypothetical": {
      "type": "object",
      "description": "What would happen outside shadow mode",
      "properties": {
        "would_allow_transition": {
          "type": "boolean"
        },
        "would_trigger_alert": {
          "type": "boolean"
        },
        "blocking_violations": {
          "type": "array",
          "items": {
            "type": "string"
          }
        }
      }
    },
    "context": {
      "type": "object",
      "description": "Additional context about the signal",
      "properties": {
        "run_id": {
          "type": "string"
        },
        "cycle": {
          "type": "integer"
        },
        "triggering_event": {
          "type": "string"
        }
      }
    }
  }
}
```

---

## Summary

This specification defines the curriculum evolution invariants for Phase X P3/P4:

1. **Monotonicity is mandatory** — Parameters can only increase, never decrease
2. **Gate thresholds can only tighten** — Coverage up, abstention down, velocity up
3. **Shadow mode is read-only** — No curriculum mutations during P3/P4 runs
4. **Drift is classified** — NONE, PARAMETRIC, SEMANTIC severity levels
5. **Violations are logged** — All events recorded to audit trail
6. **Schemas are versioned** — Timeline and governance signal formats defined

**SHADOW MODE CONTRACT MAINTAINED**: All curriculum observations are logging-only. No transitions are executed during Phase X shadow operations.

---

## 12. Phase X P5 Considerations: Real Telemetry Stress

> **STATUS**: Forward-Looking Specification
>
> This section addresses anticipated stress patterns when curriculum invariants
> are evaluated against real telemetry at P5 scale.

### 12.1 Stress Vectors Under Real Load

Phase X P5 introduces **continuous real telemetry** from live derivation runs. Unlike P3 (synthetic) and P4 (shadow coupling), P5 exercises curriculum invariants under production-like conditions:

| Stress Vector | P3/P4 Behavior | P5 Behavior | Invariant Pressure |
|---------------|----------------|-------------|-------------------|
| Transition Frequency | Rare (manual) | Frequent (gate-driven) | CUR-P4-04 under load |
| Gate Evaluation Rate | Periodic | Continuous | CUR-P3-02 threshold drift |
| Telemetry Variance | Controlled | Real noise | False positive drift signals |
| Slice Ladder Access | Read-only | Read-intensive | CUR-P3-01 consistency |
| Concurrent Observers | Single | Multiple | Snapshot coherence |

### 12.2 Anticipated Invariant Pressure Points

**CUR-P4-04 (No `activate_next_slice()` calls)**: Under P5, successful gate passage will generate frequent transition requests. The guard must distinguish between:
- Legitimate advancement signals (log and allow)
- Premature transition attempts (log and suppress)
- Concurrent advancement races (serialize or reject)

**CUR-P3-02 (Gate thresholds frozen)**: Real telemetry exhibits natural variance. Gate thresholds may appear to "drift" due to:
- Floating-point accumulation in CI calculations
- Window boundary effects in velocity measurements
- Sample size fluctuations near `sample_min` boundary

**Monotonicity under backpressure**: When derivation velocity drops, operators may be tempted to relax `depth_max` or `breadth_max` temporarily. The P5 enforcement layer must log these hypothetical violations without disrupting operations.

### 12.3 Recommended P5 Metric: Curriculum Transition Request Rate

To monitor curriculum governance health under P5 load, we propose a single summary metric:

```
METRIC: curriculum_transition_requests_per_1k_cycles (CTRPK)

Definition:
  CTRPK = (transition_requested_signals / total_cycles) × 1000

Components:
  - transition_requested_signals: Count of TRANSITION_REQUESTED governance signals
  - total_cycles: Total derivation cycles in measurement window

Measurement Window: Rolling 1-hour or per-run aggregate

Thresholds:
  - GREEN:  CTRPK < 1.0   (< 1 request per 1000 cycles)
  - YELLOW: CTRPK 1.0-5.0 (moderate transition pressure)
  - RED:    CTRPK > 5.0   (high transition churn, investigate)
```

**Rationale**: A healthy curriculum should advance infrequently—slices represent significant complexity increases. High CTRPK indicates either:
1. Gates are too permissive (thresholds need tightening)
2. Telemetry is noisy (false gate passage)
3. Slice granularity is too fine (curriculum design issue)

**Schema Extension** (for `curriculum-governance-signal/1.0.0`):

```json
{
  "p5_metrics": {
    "ctrpk": {
      "type": "number",
      "description": "Curriculum transition requests per 1000 cycles"
    },
    "measurement_window_cycles": {
      "type": "integer",
      "description": "Number of cycles in measurement window"
    },
    "transition_requests_in_window": {
      "type": "integer",
      "description": "Raw count of transition request signals"
    }
  }
}
```

### 12.4 P5 Enforcement Mode

P5 operates in **SHADOW_ACTIVE** mode:

| Mode | Observation | Logging | Blocking |
|------|-------------|---------|----------|
| P3 SHADOW | Yes | Yes | No |
| P4 SHADOW | Yes | Yes | No |
| P5 SHADOW_ACTIVE | Yes | Yes | **Advisory** |

In SHADOW_ACTIVE mode:
- All invariant violations are logged
- SEMANTIC violations emit `WOULD_BLOCK` signals
- Actual blocking is **deferred to operator decision**
- CTRPK is recorded in every governance signal

This preserves observational purity while providing actionable P5 health metrics.

---

**SHADOW MODE CONTRACT MAINTAINED**: All curriculum observations are logging-only. No transitions are executed during Phase X shadow operations.

---

## 13. Curriculum Coherence Summary

> **STATUS**: Observational Witness (Phase X)
>
> The curriculum coherence summary provides a unified view of taxonomy integrity
> across both P3 (synthetic) and P4 (real-runner shadow) experiments. This
> serves as an **alignment witness** for curriculum philosophical consistency,
> not an automatic blocker.

### 13.1 Purpose

The First Light curriculum coherence summary answers the question: **"Did the
curriculum stay philosophically consistent between synthetic and shadow modes?"**

This is a critical question for external reviewers evaluating MathLedger's
curriculum governance. The summary combines:

- **P3 Taxonomy Summary**: Alignment score and integrity status from synthetic
  experiment observations
- **P4 Taxonomy Calibration**: Drift band and projected horizon from real-runner
  shadow observations

### 13.2 Reading the Summary

The summary contains four key fields that must be interpreted together:

#### alignment_score (0.0-1.0)

Overall taxonomy alignment across metrics, documentation, and curriculum slices.
Weighted combination:
- Metrics alignment: 30% weight
- Documentation alignment: 30% weight
- Curriculum alignment: 40% weight (highest, as it affects runtime)

**Interpretation:**
- `1.0`: Perfect alignment across all systems
- `0.8-0.99`: Good alignment, minor gaps
- `0.5-0.79`: Partial alignment, review recommended
- `<0.5`: Significant misalignment, investigation required

#### integrity_status ("OK" | "WARN" | "BLOCK")

Current taxonomy integrity state derived from impact analysis.

**Interpretation:**
- `OK`: All systems aligned, no action needed
- `WARN`: Documentation or metrics out of date, review recommended
- `BLOCK`: Curriculum slices reference removed types (critical), must update
  curriculum.yaml before proceeding

#### drift_band ("STABLE" | "LOW_DRIFT" | "MEDIUM_DRIFT" | "HIGH_DRIFT")

Historical taxonomy change intensity based on drift timeline analysis.

**Interpretation:**
- `STABLE`: No taxonomy changes detected in history
- `LOW_DRIFT`: Sparse, non-breaking changes
- `MEDIUM_DRIFT`: Moderate change frequency, some breaking changes
- `HIGH_DRIFT`: Frequent breaking changes, taxonomy instability

#### projected_horizon (0.0-1.0)

Extrapolated change intensity forward, assuming current drift rate continues.

**Interpretation:**
- `0.0`: No changes projected
- `0.0-0.2`: Low projected instability
- `0.2-0.5`: Moderate projected instability
- `>0.5`: High projected instability, taxonomy may continue evolving

### 13.3 Combined Interpretation

The fields should be read together to assess curriculum coherence:

| alignment_score | integrity_status | drift_band | Interpretation |
|----------------|------------------|------------|----------------|
| High (≥0.8) | OK | STABLE | **Ideal**: Curriculum philosophically consistent between P3 and P4 |
| High (≥0.8) | OK | LOW_DRIFT | **Good**: Minor taxonomy evolution, coherence maintained |
| Medium (0.5-0.79) | WARN | MEDIUM_DRIFT | **Review**: Some misalignment, taxonomy changes may be affecting curriculum |
| Low (<0.5) | BLOCK | HIGH_DRIFT | **Critical**: Significant coherence break, curriculum slices must be updated |

### 13.4 Evidence-Only Contract

**SHADOW MODE**: The curriculum coherence summary is an **alignment witness**,
not an automatic blocker. It provides observability into curriculum coherence
but does not influence P3/P4 execution behavior.

- No blocking logic depends on coherence summary values
- No control flow is altered by summary contents
- Summary is purely for evidence pack inclusion and external review

External reviewers use this summary to:
1. Assess curriculum philosophical consistency across experiment phases
2. Identify taxonomy changes that may have broken curriculum coherence
3. Validate that curriculum governance maintained alignment during shadow operations

### 13.5 Schema

The curriculum coherence summary is included in evidence packs under:
```
evidence["governance"]["taxonomy"]["curriculum_coherence_summary"]
```

Schema version: `1.0.0`

---

## 14. Curriculum Coherence Trend

> **STATUS**: Alignment Dashboard (Phase X P5)
>
> The curriculum coherence trend provides a time-series view of alignment evolution
> across cycles or runs, enabling dashboard visualization of curriculum coherence
> over time. Used in P5 (CAL-EXP-2, 1000 cycles) to track alignment trends.

### 14.1 Purpose

The curriculum coherence time-series extracts alignment trend data from a sequence
of curriculum coherence summaries, creating a visible "Alignment Trend Line" for
dashboard visualization. This enables:

- **Trend Analysis**: Track how curriculum coherence evolves over time
- **P5 Acceptance**: Validate alignment stability across 1000-cycle experiments
- **Dashboard Integration**: Provide time-series data for CAL-EXP-* visualization

### 14.2 Time-Series Structure

The time-series is generated by `build_curriculum_coherence_timeseries()` and
contains:

```json
{
  "schema_version": "1.0.0",
  "points": [
    {
      "cycle_or_run_idx": 0,
      "alignment_score": 1.0,
      "drift_band": "STABLE"
    },
    {
      "cycle_or_run_idx": 100,
      "alignment_score": 0.95,
      "drift_band": "LOW_DRIFT"
    }
  ]
}
```

**Key Properties:**
- `cycle_or_run_idx`: Monotone increasing indices (enforced automatically)
- `alignment_score`: Alignment score at this point (0.0-1.0)
- `drift_band`: Drift classification at this point

### 14.3 Monotone Indexing

The time-series enforces **monotone increasing** indices:
- If `cycle_or_run_idx` is provided in summaries, it is used
- If missing, indices are assigned sequentially starting from 0
- Out-of-order indices are automatically corrected to maintain monotonicity

This ensures the time-series can be safely plotted as a trend line without
gaps or reversals.

### 14.4 Storage and Usage

**P5 Storage:**
- Time-series is stored under `calibration/curriculum_coherence_timeseries.json`
- Generated from coherence summaries collected during CAL-EXP-2 (1000 cycles)

**Dashboard Integration:**
- Time-series points can be plotted as:
  - X-axis: `cycle_or_run_idx`
  - Y-axis: `alignment_score` (primary trend line)
  - Color/annotation: `drift_band` (secondary signal)

**P5 Acceptance:**
- External reviewers use the trend line to assess:
  - Alignment stability over 1000 cycles
  - Whether coherence degrades during long runs
  - Correlation between drift_band changes and alignment_score

### 14.5 Cross-Check with Curriculum Governance

The time-series can be cross-checked against curriculum governance signals using
`summarize_coherence_vs_curriculum_governance()` to identify consistency, tension,
or conflict between taxonomy alignment and curriculum evolution.

#### Status Determination Rules

Status is determined per time-series point, then aggregated to worst status.
Thresholds are parameterized (defaults shown):

1. **CONSISTENT**: `drift_band ≤ MEDIUM_DRIFT AND alignment_score ≥ alignment_ok_threshold` (default: 0.8)
   - Indicates stable taxonomy with strong alignment
   - All drift bands (STABLE, LOW_DRIFT, MEDIUM_DRIFT) with high alignment

2. **TENSION**: `(drift_band == high_drift_value XOR alignment_score < alignment_conflict_threshold)` (default: HIGH_DRIFT, 0.6)
   - Covers two scenarios:
     - High drift with acceptable alignment (high_drift_value AND alignment_score ≥ alignment_conflict_threshold)
     - Low alignment with stable drift (drift_band ≤ MEDIUM_DRIFT AND alignment_score < alignment_conflict_threshold)
   - Indicates misalignment between drift and alignment signals

3. **CONFLICT**: `drift_band == high_drift_value AND alignment_score < alignment_conflict_threshold` (default: HIGH_DRIFT, 0.6)
   - Worst case: high taxonomy instability with low alignment
   - Indicates critical coherence breakdown

**Parameterization** (no gating):
- `alignment_ok_threshold` (default: 0.8): Minimum alignment score for CONSISTENT status
- `alignment_conflict_threshold` (default: 0.6): Maximum alignment score for CONFLICT status
- `high_drift_value` (default: "HIGH_DRIFT"): Drift band value that triggers HIGH_DRIFT logic

Using default parameters produces identical outputs to hardcoded values.

#### Episodes

Episodes are contiguous window ranges where status != CONSISTENT, enriched with metadata:
- **Basic fields**: `start_idx`, `end_idx`, `status`
- **Metadata fields**:
  - `point_count`: Number of time-series points in this episode
  - `max_drift_band_seen`: Highest drift band observed within the episode (HIGH_DRIFT > MEDIUM_DRIFT > LOW_DRIFT > STABLE)
  - `min_alignment_score_seen`: Lowest alignment score observed within the episode
  - `episode_severity_score`: Deterministic severity score (higher = more severe)
- Episodes are merged when consecutive points share the same status
- Episodes are separated by CONSISTENT points

**Episode Severity Score**:
The severity score is calculated deterministically using:
- Base by status: CONSISTENT=0, TENSION=50, CONFLICT=100
- Drift bump: +20 if `max_drift_band_seen == high_drift_value`
- Alignment bump: `min(30, (alignment_ok_threshold - min_alignment_score_seen) * 10)`

Higher scores indicate more severe episodes. The score is monotonic with status severity (CONFLICT > TENSION > CONSISTENT).

#### Advisory Notes

The cross-check generates 1-3 neutral advisory notes:
- Notes reference which threshold triggered the status
- Notes include counts of affected points
- Notes are descriptive, not prescriptive (advisory only)

#### Summary Block

The cross-check includes a compact summary block for downstream consumption:
- `num_points`: Total number of time-series points analyzed
- `num_episodes`: Total number of episodes detected (where status != CONSISTENT)
- `worst_status`: Worst status across all points (CONFLICT > TENSION > CONSISTENT)
- `worst_episode`: Episode with highest `episode_severity_score`, with deterministic tie-breakers

**Worst Episode Selection**:
The `worst_episode` is selected using deterministic tie-breaking:
1. **Primary**: Highest `episode_severity_score`
2. **Tie-breaker 1**: Longer duration (`end_idx - start_idx`)
3. **Tie-breaker 2**: Smaller `start_idx` (earlier in timeline)

The selected episode includes a `selected_by` field indicating which criteria
were used for selection: `["severity_score"]`, `["severity_score", "duration"]`,
or `["severity_score", "duration", "start_idx"]`.

This ensures stable, auditable selection of the worst episode even when multiple
episodes have identical severity scores.

**Severity Score Basis**:
The cross-check output includes a `severity_score_basis` block at the root level
containing:
- `status_weights`: Base scores for each status (CONSISTENT=0, TENSION=50, CONFLICT=100)
- `drift_bump`: Additional score for HIGH_DRIFT episodes (+20)
- `alignment_bump_formula`: Formula for alignment-based score adjustment
- `alignment_ok_threshold_used`: Actual threshold value used
- `high_drift_value_used`: Actual high drift value used

This metadata enables auditors to reproduce severity scores and verify calculations.

The summary block enables quick assessment of cross-check results without
iterating through all episodes.

#### Evidence Attachment

Cross-check results are attached to evidence packs under:
```
evidence["governance"]["curriculum_coherence_crosscheck"]
```

**Advisory Only**: Cross-check results are observational and do not gate P5
acceptance. They provide visibility into curriculum coherence trends but do
not block or influence experiment execution.

---

## 14. CTRPK Integration Plan: Global Health, Evidence, and Council

> **STATUS**: P5 Forward Specification
>
> This section defines how the Curriculum Transition Request Rate (CTRPK)
> metric integrates with global health tiles, evidence packs, and the
> uplift council.

### 14.1 Global Health: Curriculum Stress Console Tile

The curriculum stress tile provides real-time visibility into curriculum
governance load. It displays CTRPK alongside trend analysis and council status.

**Schema**: `schemas/curriculum/curriculum_stress_console_tile.schema.json`

**Key Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `status_light` | GREEN/YELLOW/RED | Visual indicator based on CTRPK thresholds |
| `ctrpk` | number | Current CTRPK value |
| `trend.direction` | IMPROVING/STABLE/DEGRADING | Trend over observation window |
| `council_status` | OK/WARN/BLOCK | Uplift council classification |

**Status Light Derivation**:

```
if ctrpk < 1.0:
    status_light = GREEN
elif ctrpk <= 5.0:
    status_light = YELLOW
else:
    status_light = RED
```

**Trend Derivation**:

```
delta = ctrpk_1h - ctrpk_24h
if delta < -0.5:
    direction = IMPROVING
elif delta > 0.5:
    direction = DEGRADING
else:
    direction = STABLE
```

**Example Tile Payload**:

```json
{
  "schema_version": "1.0.0",
  "tile_type": "curriculum_stress",
  "timestamp": "2025-01-15T14:30:00Z",
  "status_light": "GREEN",
  "headline": "Curriculum Stable",
  "subheadline": "CTRPK: 0.42 | 3 requests in 7142 cycles",
  "ctrpk_summary": {
    "ctrpk": 0.42,
    "transition_requests": 3,
    "total_cycles": 7142,
    "measurement_window_minutes": 60,
    "blocked_requests": 0,
    "successful_transitions": 0
  },
  "trend": {
    "direction": "STABLE",
    "ctrpk_1h": 0.42,
    "ctrpk_6h": 0.38,
    "ctrpk_24h": 0.45,
    "delta_vs_baseline": -0.08,
    "baseline_ctrpk": 0.50
  },
  "council_status": "OK",
  "phase": "P5",
  "mode": "SHADOW_ACTIVE"
}
```

### 14.2 Uplift Council: CTRPK Classification Mapping

The uplift council uses CTRPK to assess curriculum governance health as one
dimension in the multi-dimensional uplift decision.

**Classification Rules**:

| CTRPK Range | Council Status | Rationale |
|-------------|----------------|-----------|
| `< 1.0` | OK | Healthy: < 1 transition request per 1000 cycles |
| `1.0 - 5.0` | WARN | Moderate pressure: transition requests elevated |
| `> 5.0` | BLOCK | High churn: curriculum instability detected |

**Override Conditions**:

| Condition | Override | Reason |
|-----------|----------|--------|
| Any SEMANTIC violation | BLOCK | Invariant breach regardless of CTRPK |
| `blocked_requests > 0` with CTRPK < 1.0 | WARN | Hidden pressure |
| `semantic_violations > 0` | BLOCK | Monotonicity or gate regression |
| Trend DEGRADING + CTRPK > 3.0 | BLOCK | Accelerating instability |

**Council Integration Function** (pure helper):

```python
def council_classify_ctrpk(
    ctrpk: float,
    semantic_violations: int = 0,
    blocked_requests: int = 0,
    trend_direction: str = "STABLE",
) -> Literal["OK", "WARN", "BLOCK"]:
    """
    Classify CTRPK for uplift council.

    Returns: "OK", "WARN", or "BLOCK"
    """
    # Hard blocks
    if semantic_violations > 0:
        return "BLOCK"
    if ctrpk > 5.0:
        return "BLOCK"
    if trend_direction == "DEGRADING" and ctrpk > 3.0:
        return "BLOCK"

    # Warnings
    if ctrpk > 1.0:
        return "WARN"
    if blocked_requests > 0:
        return "WARN"

    return "OK"
```

**Council View Integration**:

The curriculum stress status is included in the unified council view under:
```
council_view["dimensions"]["curriculum_stress"] = {
    "status": "OK" | "WARN" | "BLOCK",
    "ctrpk": 0.42,
    "trend": "STABLE",
    "semantic_violations": 0
}
```

### 14.3 Evidence Pack: CTRPK Compact Field

CTRPK is recorded in evidence packs as a compact governance metric for
external reviewers.

**Evidence Path**:
```
evidence["governance"]["curriculum"]["ctrpk"]
```

**Compact Schema**:

```json
{
  "ctrpk": {
    "type": "object",
    "required": ["value", "status", "window_cycles"],
    "properties": {
      "value": {
        "type": "number",
        "description": "CTRPK value"
      },
      "status": {
        "type": "string",
        "enum": ["OK", "WARN", "BLOCK"],
        "description": "Council classification"
      },
      "window_cycles": {
        "type": "integer",
        "description": "Measurement window in cycles"
      },
      "transition_requests": {
        "type": "integer",
        "description": "Raw request count"
      },
      "trend": {
        "type": "string",
        "enum": ["IMPROVING", "STABLE", "DEGRADING"]
      }
    }
  }
}
```

**Example Evidence Entry**:

```json
{
  "governance": {
    "curriculum": {
      "signal": { ... },
      "timeline_summary": { ... },
      "council_status": "OK",
      "ctrpk": {
        "value": 0.42,
        "status": "OK",
        "window_cycles": 7142,
        "transition_requests": 3,
        "trend": "STABLE"
      }
    }
  }
}
```

### 14.4 Global Health Integration

The curriculum stress tile integrates with the global health surface as follows:

```
global_health_surface
├── envelope_v4
│   └── components
│       └── curriculum_stress: { band: "GREEN", ctrpk: 0.42 }
├── tiles
│   └── curriculum_stress: <full tile payload>
└── council_view
    └── dimensions
        └── curriculum_stress: { status: "OK", ctrpk: 0.42 }
```

**Band Mapping for Envelope v4**:

| CTRPK | Band |
|-------|------|
| `< 1.0` | GREEN |
| `1.0 - 5.0` | YELLOW |
| `> 5.0` | RED |

---

## 15. Smoke-Test Readiness Checklist

> **PURPOSE**: Verify CTRPK integration is complete before P5 activation.

### 15.1 Schema Readiness

| Item | File | Status |
|------|------|--------|
| Console tile schema | `schemas/curriculum/curriculum_stress_console_tile.schema.json` | Required |
| Evidence compact field | Embedded in evidence pack schema | Required |
| Council dimension schema | Part of council view | Required |

### 15.2 Implementation Readiness

| Component | Function/Class | Checklist |
|-----------|----------------|-----------|
| CTRPK Calculator | `compute_ctrpk(signals, cycles)` | [ ] Returns float |
| Tile Builder | `build_curriculum_stress_tile(ctrpk, trend, ...)` | [ ] Schema-compliant |
| Council Classifier | `council_classify_ctrpk(ctrpk, ...)` | [ ] Pure function |
| Evidence Attacher | `attach_ctrpk_to_evidence(evidence, ctrpk)` | [ ] Non-mutating |

### 15.3 Test Coverage

| Test | Description | Status |
|------|-------------|--------|
| `test_ctrpk_green_threshold` | CTRPK < 1.0 produces GREEN | Required |
| `test_ctrpk_yellow_threshold` | CTRPK 1.0-5.0 produces YELLOW | Required |
| `test_ctrpk_red_threshold` | CTRPK > 5.0 produces RED | Required |
| `test_council_semantic_override` | Semantic violations force BLOCK | Required |
| `test_trend_degrading_block` | Degrading trend + CTRPK > 3.0 blocks | Required |
| `test_evidence_attachment` | CTRPK appears in evidence pack | Required |
| `test_tile_schema_compliance` | Tile validates against schema | Required |
| `test_determinism` | Same inputs produce same outputs | Required |

### 15.4 Integration Points

| Integration | Endpoint | Verification |
|-------------|----------|--------------|
| Global Health | `envelope_v4.components.curriculum_stress` | [ ] Band present |
| Dashboard Tile | `tiles.curriculum_stress` | [ ] Renders correctly |
| Council View | `council_view.dimensions.curriculum_stress` | [ ] Status present |
| Evidence Pack | `evidence.governance.curriculum.ctrpk` | [ ] Compact field present |

### 15.5 Go/No-Go Criteria

**GO** if all:
- [ ] Schema validates with JSON Schema Draft-07
- [ ] All 8 required tests pass
- [ ] Tile renders in dashboard mock
- [ ] Evidence pack includes CTRPK compact field
- [ ] Council view shows curriculum_stress dimension

**NO-GO** if any:
- [ ] Schema validation fails
- [ ] Council classification produces incorrect status
- [ ] Trend calculation is non-deterministic
- [ ] Evidence pack missing CTRPK field

---

*Document Version: 1.3.0*
*Last Updated: 2025-12-11*
*Status: Specification*
