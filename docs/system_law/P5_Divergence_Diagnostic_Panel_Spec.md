# P5 Divergence Diagnostic Panel Specification

---

> **SYSTEM LAW DOCUMENT — P5 DIVERGENCE DIAGNOSTIC PANEL**
>
> This document specifies the diagnostic panel schema, rule engine, and
> implementation interface for P5 real-telemetry divergence interpretation.
>
> **Status**: SPECIFICATION
> **Version**: 1.0.0
> **Date**: 2025-12-11
> **Author**: CLAUDE C (Consolidation Layer)
> **Upstream**: Replay_Safety_Governance_Law.md Section 4.5

---

## Table of Contents

1. [Overview](#1-overview)
2. [Diagnostic Panel Schema](#2-diagnostic-panel-schema)
3. [Rule Engine Specification](#3-rule-engine-specification)
4. [Implementation Interface](#4-implementation-interface)
5. [Smoke-Test Readiness Checklist](#5-smoke-test-readiness-checklist)

---

## 1. Overview

### 1.1 Purpose

The P5 Divergence Diagnostic Panel provides a unified interpretation layer that:
- Synthesizes signals from replay safety, topology, budget, and divergence analysis
- Applies root cause attribution rules to identify likely failure sources
- Produces actionable diagnostic output for engineers and auditors

### 1.2 SHADOW MODE CONTRACT

All P5 diagnostic output is **observational only**:
- Diagnostics do NOT trigger remediation
- No control flow depends on diagnostic results
- Output is for logging, analysis, and evidence collection

### 1.3 RECONCILIATION VIEW DISCLAIMER

> **P5 Diagnostic is a RECONCILIATION VIEW; NOT a metric authority.**
>
> The P5 diagnostic panel reports exactly which signals were consumed and their
> sources (real, stub, none). It does NOT define or own the metrics it reports.
> Canonical metric definitions are maintained in:
> `docs/system_law/calibration/METRIC_DEFINITIONS.md`
>
> **Key invariants**:
> - All source labels are canonical: `{"real", "stub", "none"}`
> - Unknown sources are coerced to "stub" with advisory notes
> - All lists are deterministically sorted (alphabetical by signal name)
> - All lists are capped to declared maximums (see `_build_signal_inputs()`)
> - The `diagnostic_integrity` block provides transparency on data quality
>
> **FINAL LOCK — Usage Constraints**:
> The `diagnostic_integrity` fields (including `uses_only_canonical_sources`,
> `coerced_sources_count`, `missing_required_count`) are NOT metric authority—they
> report data quality observations only. The `reason_codes_top3` field is provided
> for **triage purposes only**: it helps engineers prioritize investigation but
> carries no governance weight. **P5 diagnostic outputs MUST NEVER be used for
> gating decisions.** All outputs remain observational under SHADOW MODE.
>
> Enforcement: this constraint is validated by tests in `tests/health/test_p5_diagnostic_harness_integration.py`.
> Run: `pytest tests/health/test_p5_diagnostic_harness_integration.py::TestNeverLieAuditCaps -v`
> Note: tests enforce non-gating semantics, not correctness of downstream governance.
>
> This panel is used for CAL-EXP instrument calibration and triage only; it is not part of divergence minimization acceptance criteria.

### 1.4 Signal Precedence (from Replay_Safety_Governance_Law.md 4.5.4)

```
Priority 1: Identity Signal (cryptographic)     → Security-critical
Priority 2: Structure Signal (DAG coherence)    → Data integrity
Priority 3: Replay Safety Signal                → Determinism verification
Priority 4: Topology Signal                     → Model/manifold health
Priority 5: Budget Signal                       → Resource constraints
Priority 6: Metrics Signal                      → Performance indicators
```

---

## 2. Diagnostic Panel Schema

### 2.1 Schema Definition

```json
// # SPEC-ONLY — p5_divergence_diagnostic.schema.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://mathledger.org/schemas/p5/divergence_diagnostic.v1.0.0.json",
  "title": "P5 Divergence Diagnostic Panel",
  "description": "Unified diagnostic output for P5 real-telemetry divergence interpretation",
  "version": "1.0.0",
  "type": "object",
  "required": [
    "schema_version",
    "timestamp",
    "cycle",
    "replay_status",
    "divergence_severity",
    "divergence_type",
    "topology_mode",
    "budget_stability",
    "root_cause_hypothesis",
    "supporting_signals",
    "action"
  ],
  "properties": {
    "schema_version": {
      "type": "string",
      "const": "1.0.0"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time"
    },
    "cycle": {
      "type": "integer",
      "minimum": 0
    },
    "run_id": {
      "type": "string",
      "description": "Optional run identifier for traceability"
    },

    "replay_status": {
      "type": "string",
      "enum": ["OK", "WARN", "BLOCK"],
      "description": "Replay safety governance status"
    },
    "divergence_severity": {
      "type": "string",
      "enum": ["NONE", "INFO", "WARN", "CRITICAL"],
      "description": "P4 divergence severity classification"
    },
    "divergence_type": {
      "type": "string",
      "enum": ["NONE", "STATE", "OUTCOME", "COMBINED"],
      "description": "Type of divergence detected"
    },
    "topology_mode": {
      "type": "string",
      "enum": ["STABLE", "DRIFT", "TURBULENT", "CRITICAL"],
      "description": "Topology bundle mode classification"
    },
    "budget_stability": {
      "type": "string",
      "enum": ["STABLE", "DRIFTING", "VOLATILE"],
      "description": "Budget stability classification"
    },

    "root_cause_hypothesis": {
      "type": "string",
      "enum": [
        "NOMINAL",
        "REPLAY_FAILURE",
        "STRUCTURAL_BREAK",
        "PHASE_LAG",
        "BUDGET_CONFOUND",
        "IDENTITY_VIOLATION",
        "STRUCTURE_VIOLATION",
        "CASCADING_FAILURE",
        "UNKNOWN"
      ],
      "description": "Primary root cause attribution"
    },
    "root_cause_confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Confidence in root cause attribution"
    },

    "supporting_signals": {
      "type": "object",
      "description": "Evidence supporting the diagnosis",
      "properties": {
        "replay": {
          "type": "object",
          "properties": {
            "status": { "type": "string" },
            "alignment": { "type": "string" },
            "conflict": { "type": "boolean" },
            "reasons": { "type": "array", "items": { "type": "string" } }
          }
        },
        "topology": {
          "type": "object",
          "properties": {
            "mode": { "type": "string" },
            "persistence_drift": { "type": "number" },
            "betti_0": { "type": "integer" },
            "betti_1": { "type": "integer" },
            "within_omega": { "type": "boolean" }
          }
        },
        "budget": {
          "type": "object",
          "properties": {
            "stability_class": { "type": "string" },
            "health_score": { "type": "number" },
            "stability_index": { "type": "number" }
          }
        },
        "divergence": {
          "type": "object",
          "properties": {
            "severity": { "type": "string" },
            "type": { "type": "string" },
            "divergence_pct": { "type": "number" },
            "H_diff": { "type": "number" },
            "rho_diff": { "type": "number" }
          }
        },
        "identity": {
          "type": "object",
          "properties": {
            "status": { "type": "string" },
            "hash_valid": { "type": "boolean" },
            "chain_continuous": { "type": "boolean" }
          }
        },
        "structure": {
          "type": "object",
          "properties": {
            "status": { "type": "string" },
            "dag_coherent": { "type": "boolean" },
            "cycle_detected": { "type": "boolean" }
          }
        }
      }
    },

    "attribution_chain": {
      "type": "array",
      "description": "Ordered list of signals considered in attribution",
      "items": {
        "type": "object",
        "properties": {
          "signal": { "type": "string" },
          "status": { "type": "string" },
          "contributed_to_diagnosis": { "type": "boolean" },
          "reason": { "type": "string" }
        }
      }
    },

    "confounding_factors": {
      "type": "array",
      "description": "Factors that reduce confidence in root cause",
      "items": { "type": "string" }
    },

    "action": {
      "type": "string",
      "enum": ["LOGGED_ONLY", "REVIEW_RECOMMENDED", "INVESTIGATION_REQUIRED"],
      "description": "Recommended action level (advisory only)"
    },
    "action_rationale": {
      "type": "string",
      "description": "Human-readable explanation of action recommendation"
    },

    "headline": {
      "type": "string",
      "description": "One-line summary for dashboard display"
    }
  },
  "additionalProperties": false
}
```

### 2.2 Example Output

```json
// # SPEC-ONLY — Example: STRUCTURAL_BREAK diagnosis
{
  "schema_version": "1.0.0",
  "timestamp": "2025-12-11T10:30:00Z",
  "cycle": 1042,
  "run_id": "p5_20251211_103000",

  "replay_status": "OK",
  "divergence_severity": "WARN",
  "divergence_type": "STATE",
  "topology_mode": "TURBULENT",
  "budget_stability": "STABLE",

  "root_cause_hypothesis": "STRUCTURAL_BREAK",
  "root_cause_confidence": 0.85,

  "supporting_signals": {
    "replay": {
      "status": "OK",
      "alignment": "aligned",
      "conflict": false,
      "reasons": ["[Safety] All checks passed"]
    },
    "topology": {
      "mode": "TURBULENT",
      "persistence_drift": 0.18,
      "betti_0": 1,
      "betti_1": 2,
      "within_omega": true
    },
    "budget": {
      "stability_class": "STABLE",
      "health_score": 92,
      "stability_index": 0.96
    },
    "divergence": {
      "severity": "WARN",
      "type": "STATE",
      "divergence_pct": 0.12,
      "H_diff": 0.08,
      "rho_diff": 0.05
    }
  },

  "attribution_chain": [
    {"signal": "identity", "status": "OK", "contributed_to_diagnosis": false, "reason": "Cryptographic integrity intact"},
    {"signal": "structure", "status": "OK", "contributed_to_diagnosis": false, "reason": "DAG coherent"},
    {"signal": "replay", "status": "OK", "contributed_to_diagnosis": true, "reason": "Determinism confirmed—rules out replay bug"},
    {"signal": "topology", "status": "TURBULENT", "contributed_to_diagnosis": true, "reason": "Primary signal—structural regime shift"},
    {"signal": "budget", "status": "STABLE", "contributed_to_diagnosis": false, "reason": "No confounding"},
    {"signal": "metrics", "status": "OK", "contributed_to_diagnosis": false, "reason": "Performance nominal"}
  ],

  "confounding_factors": [],

  "action": "REVIEW_RECOMMENDED",
  "action_rationale": "Topology turbulence with STATE divergence indicates model recalibration may be needed",

  "headline": "STRUCTURAL_BREAK: Topology TURBULENT, Replay OK — Model/manifold shift detected"
}
```

---

## 3. Rule Engine Specification

### 3.1 Evaluation Order

The rule engine processes signals in strict precedence order. Higher-priority failures short-circuit evaluation.

```python
# # SPEC-ONLY — Rule Engine Pseudocode

def evaluate_divergence_diagnostic(
    identity_signal: Optional[Dict],
    structure_signal: Optional[Dict],
    replay_signal: Dict,
    topology_signal: Dict,
    budget_signal: Dict,
    divergence_snapshot: Dict,
) -> DiagnosticResult:
    """
    Evaluate divergence diagnostic using precedence-ordered rules.

    SHADOW MODE: Output is observational only.
    """

    # Initialize attribution chain
    attribution_chain = []
    confounding_factors = []

    # =========================================================================
    # PHASE 1: Check high-priority security/integrity signals
    # =========================================================================

    # Priority 1: Identity Signal (cryptographic)
    if identity_signal:
        identity_status = evaluate_identity(identity_signal)
        attribution_chain.append({
            "signal": "identity",
            "status": identity_status,
            "contributed_to_diagnosis": identity_status == "BLOCK",
            "reason": get_identity_reason(identity_signal)
        })

        if identity_status == "BLOCK":
            return DiagnosticResult(
                root_cause_hypothesis="IDENTITY_VIOLATION",
                confidence=1.0,
                action="INVESTIGATION_REQUIRED",
                headline="IDENTITY_VIOLATION: Cryptographic integrity compromised"
            )

    # Priority 2: Structure Signal (DAG coherence)
    if structure_signal:
        structure_status = evaluate_structure(structure_signal)
        attribution_chain.append({
            "signal": "structure",
            "status": structure_status,
            "contributed_to_diagnosis": structure_status == "BLOCK",
            "reason": get_structure_reason(structure_signal)
        })

        if structure_status == "BLOCK":
            return DiagnosticResult(
                root_cause_hypothesis="STRUCTURE_VIOLATION",
                confidence=1.0,
                action="INVESTIGATION_REQUIRED",
                headline="STRUCTURE_VIOLATION: DAG coherence broken"
            )

    # =========================================================================
    # PHASE 2: Extract core signal values
    # =========================================================================

    replay_status = replay_signal.get("status", "OK").upper()
    topology_mode = topology_signal.get("mode", "STABLE").upper()
    budget_stability = budget_signal.get("stability_class", "STABLE").upper()
    divergence_severity = divergence_snapshot.get("severity", "NONE").upper()
    divergence_type = divergence_snapshot.get("type", "NONE").upper()

    # =========================================================================
    # PHASE 3: Check for confounding factors
    # =========================================================================

    if budget_stability == "VOLATILE":
        confounding_factors.append("budget_volatile")
    elif budget_stability == "DRIFTING":
        confounding_factors.append("budget_drifting")

    if topology_mode in ("TURBULENT", "CRITICAL"):
        confounding_factors.append("topology_unstable")

    # =========================================================================
    # PHASE 4: Apply diagnostic rules
    # =========================================================================

    # Rule 1: NOMINAL — All systems healthy
    if (replay_status == "OK" and
        topology_mode == "STABLE" and
        budget_stability == "STABLE" and
        divergence_severity in ("NONE", "INFO")):

        return DiagnosticResult(
            root_cause_hypothesis="NOMINAL",
            confidence=1.0,
            action="LOGGED_ONLY",
            headline="NOMINAL: All systems healthy"
        )

    # Rule 2: REPLAY_FAILURE — Replay failed in stable environment
    if (replay_status == "BLOCK" and
        topology_mode == "STABLE" and
        budget_stability == "STABLE"):

        return DiagnosticResult(
            root_cause_hypothesis="REPLAY_FAILURE",
            confidence=0.95,
            action="INVESTIGATION_REQUIRED",
            headline="REPLAY_FAILURE: Determinism bug in stable environment"
        )

    # Rule 3: STRUCTURAL_BREAK — Replay OK but topology turbulent/critical
    if (replay_status == "OK" and
        topology_mode in ("TURBULENT", "CRITICAL") and
        divergence_type == "STATE"):

        return DiagnosticResult(
            root_cause_hypothesis="STRUCTURAL_BREAK",
            confidence=0.85,
            action="REVIEW_RECOMMENDED",
            headline=f"STRUCTURAL_BREAK: Topology {topology_mode}, Replay OK — Model shift"
        )

    # Rule 4: PHASE_LAG — Multiple systems showing correlated strain
    if (replay_status == "WARN" and
        topology_mode == "DRIFT" and
        budget_stability in ("DRIFTING", "VOLATILE")):

        return DiagnosticResult(
            root_cause_hypothesis="PHASE_LAG",
            confidence=0.70,
            action="REVIEW_RECOMMENDED",
            confounding_factors=confounding_factors,
            headline="PHASE_LAG: Correlated multi-system drift"
        )

    # Rule 5: BUDGET_CONFOUND — Divergence during budget instability
    if (budget_stability == "VOLATILE" and
        divergence_severity in ("WARN", "CRITICAL")):

        return DiagnosticResult(
            root_cause_hypothesis="BUDGET_CONFOUND",
            confidence=0.60,
            action="REVIEW_RECOMMENDED",
            confounding_factors=["budget_volatile"],
            headline="BUDGET_CONFOUND: Divergence during budget instability"
        )

    # Rule 6: CASCADING_FAILURE — Multiple high-severity signals
    block_count = sum([
        1 if replay_status == "BLOCK" else 0,
        1 if topology_mode == "CRITICAL" else 0,
        1 if divergence_severity == "CRITICAL" else 0,
    ])

    if block_count >= 2:
        return DiagnosticResult(
            root_cause_hypothesis="CASCADING_FAILURE",
            confidence=0.75,
            action="INVESTIGATION_REQUIRED",
            headline="CASCADING_FAILURE: Multiple systems in critical state"
        )

    # Rule 7: Replay WARN with stable environment — potential issue
    if replay_status == "WARN" and topology_mode == "STABLE":
        return DiagnosticResult(
            root_cause_hypothesis="REPLAY_FAILURE",
            confidence=0.70,
            action="REVIEW_RECOMMENDED",
            headline="REPLAY_FAILURE: Minor replay concern in stable environment"
        )

    # Rule 8: Default — Unknown pattern
    return DiagnosticResult(
        root_cause_hypothesis="UNKNOWN",
        confidence=0.50,
        action="REVIEW_RECOMMENDED",
        confounding_factors=confounding_factors,
        headline=f"UNKNOWN: Replay={replay_status}, Topology={topology_mode}, Budget={budget_stability}"
    )
```

### 3.2 Rule Summary Table

| Rule | Condition | Hypothesis | Confidence | Action |
|------|-----------|------------|------------|--------|
| 1 | All OK, divergence NONE/INFO | NOMINAL | 1.00 | LOGGED_ONLY |
| 2 | Replay BLOCK, Topology STABLE, Budget STABLE | REPLAY_FAILURE | 0.95 | INVESTIGATION_REQUIRED |
| 3 | Replay OK, Topology TURBULENT/CRITICAL, Type STATE | STRUCTURAL_BREAK | 0.85 | REVIEW_RECOMMENDED |
| 4 | Replay WARN, Topology DRIFT, Budget DRIFTING/VOLATILE | PHASE_LAG | 0.70 | REVIEW_RECOMMENDED |
| 5 | Budget VOLATILE, Divergence WARN/CRITICAL | BUDGET_CONFOUND | 0.60 | REVIEW_RECOMMENDED |
| 6 | 2+ BLOCK/CRITICAL signals | CASCADING_FAILURE | 0.75 | INVESTIGATION_REQUIRED |
| 7 | Replay WARN, Topology STABLE | REPLAY_FAILURE | 0.70 | REVIEW_RECOMMENDED |
| 8 | Default | UNKNOWN | 0.50 | REVIEW_RECOMMENDED |

### 3.3 Hypothesis Definitions

| Hypothesis | Definition | Primary Indicator |
|------------|------------|-------------------|
| **NOMINAL** | All systems operating within normal parameters | All signals OK/INFO |
| **REPLAY_FAILURE** | Non-deterministic replay in stable environment | Replay BLOCK/WARN + stable topology/budget |
| **STRUCTURAL_BREAK** | Topology regime shift, execution deterministic | Replay OK + Topology TURBULENT/CRITICAL |
| **PHASE_LAG** | Multiple systems drifting in correlation | Replay WARN + Topology DRIFT + Budget DRIFTING |
| **BUDGET_CONFOUND** | Budget instability obscures root cause | Budget VOLATILE + significant divergence |
| **IDENTITY_VIOLATION** | Cryptographic integrity failure | Identity signal BLOCK |
| **STRUCTURE_VIOLATION** | DAG coherence broken | Structure signal BLOCK |
| **CASCADING_FAILURE** | Multiple subsystems in critical state | 2+ signals at BLOCK/CRITICAL |
| **UNKNOWN** | Pattern not recognized | No rule matched |

---

## 4. Implementation Interface

### 4.1 Core Classes

```python
# # REAL-READY — backend/analytics/p5_divergence_interpreter.py

"""
P5 Divergence Interpreter — Real-telemetry divergence diagnosis.

SHADOW MODE CONTRACT:
- All output is observational only
- No control flow depends on diagnostic results
- Output is for logging, analysis, and evidence collection

Upstream Spec: docs/system_law/P5_Divergence_Diagnostic_Panel_Spec.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class RootCauseHypothesis(str, Enum):
    """Root cause attribution hypotheses."""
    NOMINAL = "NOMINAL"
    REPLAY_FAILURE = "REPLAY_FAILURE"
    STRUCTURAL_BREAK = "STRUCTURAL_BREAK"
    PHASE_LAG = "PHASE_LAG"
    BUDGET_CONFOUND = "BUDGET_CONFOUND"
    IDENTITY_VIOLATION = "IDENTITY_VIOLATION"
    STRUCTURE_VIOLATION = "STRUCTURE_VIOLATION"
    CASCADING_FAILURE = "CASCADING_FAILURE"
    UNKNOWN = "UNKNOWN"


class DiagnosticAction(str, Enum):
    """Recommended action levels (advisory only)."""
    LOGGED_ONLY = "LOGGED_ONLY"
    REVIEW_RECOMMENDED = "REVIEW_RECOMMENDED"
    INVESTIGATION_REQUIRED = "INVESTIGATION_REQUIRED"


@dataclass
class AttributionStep:
    """Single step in the attribution chain."""
    signal: str
    status: str
    contributed_to_diagnosis: bool
    reason: str


@dataclass
class DiagnosticResult:
    """Result of divergence diagnostic evaluation."""
    root_cause_hypothesis: RootCauseHypothesis
    root_cause_confidence: float
    action: DiagnosticAction
    action_rationale: str
    headline: str
    attribution_chain: List[AttributionStep] = field(default_factory=list)
    confounding_factors: List[str] = field(default_factory=list)


@dataclass
class SupportingSignals:
    """Collected signals supporting the diagnosis."""
    replay: Optional[Dict[str, Any]] = None
    topology: Optional[Dict[str, Any]] = None
    budget: Optional[Dict[str, Any]] = None
    divergence: Optional[Dict[str, Any]] = None
    identity: Optional[Dict[str, Any]] = None
    structure: Optional[Dict[str, Any]] = None


class P5DivergenceInterpreter:
    """
    Interprets P5 divergence by synthesizing replay, topology, budget, and
    divergence signals into a unified diagnostic.

    SHADOW MODE CONTRACT:
    - interpret_divergence() returns observational data only
    - No control flow or enforcement depends on output
    - All output is for logging and analysis

    Usage:
        interpreter = P5DivergenceInterpreter()
        diagnostic = interpreter.interpret_divergence(
            divergence_snapshot=snapshot,
            replay_signal=replay,
            topology_signal=topology,
            budget_signal=budget,
        )
    """

    SCHEMA_VERSION = "1.0.0"

    def __init__(
        self,
        enable_identity_check: bool = True,
        enable_structure_check: bool = True,
    ) -> None:
        """
        Initialize the interpreter.

        Args:
            enable_identity_check: Whether to check identity signals (if provided)
            enable_structure_check: Whether to check structure signals (if provided)
        """
        self._enable_identity_check = enable_identity_check
        self._enable_structure_check = enable_structure_check

    def interpret_divergence(
        self,
        divergence_snapshot: Dict[str, Any],
        replay_signal: Dict[str, Any],
        topology_signal: Dict[str, Any],
        budget_signal: Dict[str, Any],
        *,
        identity_signal: Optional[Dict[str, Any]] = None,
        structure_signal: Optional[Dict[str, Any]] = None,
        cycle: int = 0,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Interpret divergence and produce diagnostic panel output.

        SHADOW MODE: Output is observational only.

        Args:
            divergence_snapshot: P4 divergence snapshot with severity, type, divergence_pct
            replay_signal: Replay safety governance signal with status, alignment, conflict
            topology_signal: Topology bundle signal with mode, persistence_drift, betti stats
            budget_signal: Budget signal with stability_class, health_score
            identity_signal: Optional identity signal for security checks
            structure_signal: Optional structure signal for DAG coherence checks
            cycle: Current cycle number
            run_id: Optional run identifier

        Returns:
            Dict conforming to p5_divergence_diagnostic.schema.json
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Collect supporting signals
        supporting = self._collect_supporting_signals(
            replay_signal, topology_signal, budget_signal,
            divergence_snapshot, identity_signal, structure_signal
        )

        # Run diagnostic evaluation
        result = self._evaluate(
            identity_signal=identity_signal,
            structure_signal=structure_signal,
            replay_signal=replay_signal,
            topology_signal=topology_signal,
            budget_signal=budget_signal,
            divergence_snapshot=divergence_snapshot,
        )

        # Extract status values for output
        replay_status = self._normalize_status(replay_signal.get("status", "OK"))
        divergence_severity = divergence_snapshot.get("severity", "NONE").upper()
        divergence_type = divergence_snapshot.get("type", "NONE").upper()
        topology_mode = topology_signal.get("mode", "STABLE").upper()
        budget_stability = budget_signal.get("stability_class", "STABLE").upper()

        # Build output
        output: Dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "timestamp": timestamp,
            "cycle": cycle,
            "replay_status": replay_status,
            "divergence_severity": divergence_severity,
            "divergence_type": divergence_type,
            "topology_mode": topology_mode,
            "budget_stability": budget_stability,
            "root_cause_hypothesis": result.root_cause_hypothesis.value,
            "root_cause_confidence": result.root_cause_confidence,
            "supporting_signals": supporting,
            "attribution_chain": [
                {
                    "signal": step.signal,
                    "status": step.status,
                    "contributed_to_diagnosis": step.contributed_to_diagnosis,
                    "reason": step.reason,
                }
                for step in result.attribution_chain
            ],
            "confounding_factors": result.confounding_factors,
            "action": result.action.value,
            "action_rationale": result.action_rationale,
            "headline": result.headline,
        }

        if run_id:
            output["run_id"] = run_id

        return output

    def _evaluate(
        self,
        identity_signal: Optional[Dict[str, Any]],
        structure_signal: Optional[Dict[str, Any]],
        replay_signal: Dict[str, Any],
        topology_signal: Dict[str, Any],
        budget_signal: Dict[str, Any],
        divergence_snapshot: Dict[str, Any],
    ) -> DiagnosticResult:
        """
        Core evaluation logic implementing the rule engine.

        See Section 3.1 of P5_Divergence_Diagnostic_Panel_Spec.md
        """
        attribution_chain: List[AttributionStep] = []
        confounding_factors: List[str] = []

        # Phase 1: High-priority security/integrity signals
        if self._enable_identity_check and identity_signal:
            identity_status = self._evaluate_identity(identity_signal)
            attribution_chain.append(AttributionStep(
                signal="identity",
                status=identity_status,
                contributed_to_diagnosis=(identity_status == "BLOCK"),
                reason=self._get_identity_reason(identity_signal, identity_status),
            ))

            if identity_status == "BLOCK":
                return DiagnosticResult(
                    root_cause_hypothesis=RootCauseHypothesis.IDENTITY_VIOLATION,
                    root_cause_confidence=1.0,
                    action=DiagnosticAction.INVESTIGATION_REQUIRED,
                    action_rationale="Cryptographic integrity failure detected",
                    headline="IDENTITY_VIOLATION: Cryptographic integrity compromised",
                    attribution_chain=attribution_chain,
                )

        if self._enable_structure_check and structure_signal:
            structure_status = self._evaluate_structure(structure_signal)
            attribution_chain.append(AttributionStep(
                signal="structure",
                status=structure_status,
                contributed_to_diagnosis=(structure_status == "BLOCK"),
                reason=self._get_structure_reason(structure_signal, structure_status),
            ))

            if structure_status == "BLOCK":
                return DiagnosticResult(
                    root_cause_hypothesis=RootCauseHypothesis.STRUCTURE_VIOLATION,
                    root_cause_confidence=1.0,
                    action=DiagnosticAction.INVESTIGATION_REQUIRED,
                    action_rationale="DAG coherence or cycle detection failure",
                    headline="STRUCTURE_VIOLATION: DAG coherence broken",
                    attribution_chain=attribution_chain,
                )

        # Phase 2: Extract core values
        replay_status = self._normalize_status(replay_signal.get("status", "OK"))
        topology_mode = topology_signal.get("mode", "STABLE").upper()
        budget_stability = budget_signal.get("stability_class", "STABLE").upper()
        divergence_severity = divergence_snapshot.get("severity", "NONE").upper()
        divergence_type = divergence_snapshot.get("type", "NONE").upper()

        # Phase 3: Check confounding factors
        if budget_stability == "VOLATILE":
            confounding_factors.append("budget_volatile")
        elif budget_stability == "DRIFTING":
            confounding_factors.append("budget_drifting")

        if topology_mode in ("TURBULENT", "CRITICAL"):
            confounding_factors.append("topology_unstable")

        # Add remaining signals to attribution chain
        attribution_chain.append(AttributionStep(
            signal="replay",
            status=replay_status,
            contributed_to_diagnosis=False,  # Updated by rules
            reason=self._get_replay_reason(replay_signal, replay_status),
        ))
        attribution_chain.append(AttributionStep(
            signal="topology",
            status=topology_mode,
            contributed_to_diagnosis=False,
            reason=f"Mode: {topology_mode}",
        ))
        attribution_chain.append(AttributionStep(
            signal="budget",
            status=budget_stability,
            contributed_to_diagnosis=False,
            reason=f"Stability: {budget_stability}",
        ))
        attribution_chain.append(AttributionStep(
            signal="divergence",
            status=divergence_severity,
            contributed_to_diagnosis=False,
            reason=f"Severity: {divergence_severity}, Type: {divergence_type}",
        ))

        # Phase 4: Apply diagnostic rules

        # Rule 1: NOMINAL
        if (replay_status == "OK" and
            topology_mode == "STABLE" and
            budget_stability == "STABLE" and
            divergence_severity in ("NONE", "INFO")):

            return DiagnosticResult(
                root_cause_hypothesis=RootCauseHypothesis.NOMINAL,
                root_cause_confidence=1.0,
                action=DiagnosticAction.LOGGED_ONLY,
                action_rationale="All systems operating within normal parameters",
                headline="NOMINAL: All systems healthy",
                attribution_chain=attribution_chain,
            )

        # Rule 2: REPLAY_FAILURE (strict)
        if (replay_status == "BLOCK" and
            topology_mode == "STABLE" and
            budget_stability == "STABLE"):

            self._mark_contributed(attribution_chain, "replay")
            return DiagnosticResult(
                root_cause_hypothesis=RootCauseHypothesis.REPLAY_FAILURE,
                root_cause_confidence=0.95,
                action=DiagnosticAction.INVESTIGATION_REQUIRED,
                action_rationale="Replay failure in stable environment indicates determinism bug",
                headline="REPLAY_FAILURE: Determinism bug in stable environment",
                attribution_chain=attribution_chain,
            )

        # Rule 3: STRUCTURAL_BREAK
        if (replay_status == "OK" and
            topology_mode in ("TURBULENT", "CRITICAL") and
            divergence_type == "STATE"):

            self._mark_contributed(attribution_chain, "replay")
            self._mark_contributed(attribution_chain, "topology")
            return DiagnosticResult(
                root_cause_hypothesis=RootCauseHypothesis.STRUCTURAL_BREAK,
                root_cause_confidence=0.85,
                action=DiagnosticAction.REVIEW_RECOMMENDED,
                action_rationale="Topology turbulence with STATE divergence indicates model shift",
                headline=f"STRUCTURAL_BREAK: Topology {topology_mode}, Replay OK — Model shift",
                attribution_chain=attribution_chain,
            )

        # Rule 4: PHASE_LAG
        if (replay_status == "WARN" and
            topology_mode == "DRIFT" and
            budget_stability in ("DRIFTING", "VOLATILE")):

            self._mark_contributed(attribution_chain, "replay")
            self._mark_contributed(attribution_chain, "topology")
            self._mark_contributed(attribution_chain, "budget")
            return DiagnosticResult(
                root_cause_hypothesis=RootCauseHypothesis.PHASE_LAG,
                root_cause_confidence=0.70,
                action=DiagnosticAction.REVIEW_RECOMMENDED,
                action_rationale="Multiple systems showing correlated drift",
                headline="PHASE_LAG: Correlated multi-system drift",
                attribution_chain=attribution_chain,
                confounding_factors=confounding_factors,
            )

        # Rule 5: BUDGET_CONFOUND
        if (budget_stability == "VOLATILE" and
            divergence_severity in ("WARN", "CRITICAL")):

            self._mark_contributed(attribution_chain, "budget")
            self._mark_contributed(attribution_chain, "divergence")
            return DiagnosticResult(
                root_cause_hypothesis=RootCauseHypothesis.BUDGET_CONFOUND,
                root_cause_confidence=0.60,
                action=DiagnosticAction.REVIEW_RECOMMENDED,
                action_rationale="Budget instability may be obscuring true root cause",
                headline="BUDGET_CONFOUND: Divergence during budget instability",
                attribution_chain=attribution_chain,
                confounding_factors=["budget_volatile"],
            )

        # Rule 6: CASCADING_FAILURE
        block_count = sum([
            1 if replay_status == "BLOCK" else 0,
            1 if topology_mode == "CRITICAL" else 0,
            1 if divergence_severity == "CRITICAL" else 0,
        ])

        if block_count >= 2:
            return DiagnosticResult(
                root_cause_hypothesis=RootCauseHypothesis.CASCADING_FAILURE,
                root_cause_confidence=0.75,
                action=DiagnosticAction.INVESTIGATION_REQUIRED,
                action_rationale="Multiple subsystems in critical state",
                headline="CASCADING_FAILURE: Multiple systems in critical state",
                attribution_chain=attribution_chain,
                confounding_factors=confounding_factors,
            )

        # Rule 7: REPLAY_FAILURE (soft)
        if replay_status == "WARN" and topology_mode == "STABLE":
            self._mark_contributed(attribution_chain, "replay")
            return DiagnosticResult(
                root_cause_hypothesis=RootCauseHypothesis.REPLAY_FAILURE,
                root_cause_confidence=0.70,
                action=DiagnosticAction.REVIEW_RECOMMENDED,
                action_rationale="Minor replay concern warrants investigation",
                headline="REPLAY_FAILURE: Minor replay concern in stable environment",
                attribution_chain=attribution_chain,
            )

        # Rule 8: UNKNOWN
        return DiagnosticResult(
            root_cause_hypothesis=RootCauseHypothesis.UNKNOWN,
            root_cause_confidence=0.50,
            action=DiagnosticAction.REVIEW_RECOMMENDED,
            action_rationale="Pattern not recognized by rule engine",
            headline=f"UNKNOWN: Replay={replay_status}, Topology={topology_mode}, Budget={budget_stability}",
            attribution_chain=attribution_chain,
            confounding_factors=confounding_factors,
        )

    def _normalize_status(self, status: Any) -> str:
        """Normalize status to uppercase string."""
        if hasattr(status, "value"):
            return str(status.value).upper()
        return str(status).upper()

    def _evaluate_identity(self, signal: Dict[str, Any]) -> str:
        """Evaluate identity signal status."""
        if not signal.get("block_hash_valid", True):
            return "BLOCK"
        if not signal.get("chain_continuous", True):
            return "BLOCK"
        if not signal.get("merkle_root_valid", True):
            return "BLOCK"
        if not signal.get("dual_root_consistent", True):
            return "BLOCK"
        return "OK"

    def _evaluate_structure(self, signal: Dict[str, Any]) -> str:
        """Evaluate structure signal status."""
        if not signal.get("dag_coherent", True):
            return "BLOCK"
        if signal.get("cycle_detected", False):
            return "BLOCK"
        return "OK"

    def _get_identity_reason(self, signal: Dict[str, Any], status: str) -> str:
        """Get reason for identity status."""
        if status == "BLOCK":
            if not signal.get("block_hash_valid", True):
                return "Block hash validation failed"
            if not signal.get("chain_continuous", True):
                return "Chain continuity broken"
            if not signal.get("merkle_root_valid", True):
                return "Merkle root invalid"
            return "Identity violation detected"
        return "Cryptographic integrity intact"

    def _get_structure_reason(self, signal: Dict[str, Any], status: str) -> str:
        """Get reason for structure status."""
        if status == "BLOCK":
            if not signal.get("dag_coherent", True):
                return "DAG coherence check failed"
            if signal.get("cycle_detected", False):
                return "Cycle detected in DAG"
            return "Structure violation detected"
        return "DAG coherent"

    def _get_replay_reason(self, signal: Dict[str, Any], status: str) -> str:
        """Get reason for replay status."""
        reasons = signal.get("reasons", [])
        if reasons:
            return reasons[0] if len(reasons) == 1 else f"{len(reasons)} issues"
        if status == "OK":
            return "Determinism confirmed"
        if status == "WARN":
            return "Minor replay concern"
        return "Replay verification failed"

    def _collect_supporting_signals(
        self,
        replay: Dict[str, Any],
        topology: Dict[str, Any],
        budget: Dict[str, Any],
        divergence: Dict[str, Any],
        identity: Optional[Dict[str, Any]],
        structure: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Collect supporting signals for output."""
        supporting: Dict[str, Any] = {}

        supporting["replay"] = {
            "status": self._normalize_status(replay.get("status", "OK")),
            "alignment": str(replay.get("governance_alignment", replay.get("alignment", "aligned"))).lower(),
            "conflict": replay.get("conflict", False),
            "reasons": replay.get("reasons", []),
        }

        supporting["topology"] = {
            "mode": topology.get("mode", "STABLE").upper(),
            "persistence_drift": topology.get("persistence_drift"),
            "betti_0": topology.get("betti_0") or topology.get("betti", {}).get("b0"),
            "betti_1": topology.get("betti_1") or topology.get("betti", {}).get("b1"),
            "within_omega": topology.get("within_omega"),
        }

        supporting["budget"] = {
            "stability_class": budget.get("stability_class", "STABLE").upper(),
            "health_score": budget.get("health_score"),
            "stability_index": budget.get("stability_index"),
        }

        supporting["divergence"] = {
            "severity": divergence.get("severity", "NONE").upper(),
            "type": divergence.get("type", "NONE").upper(),
            "divergence_pct": divergence.get("divergence_pct"),
            "H_diff": divergence.get("H_diff"),
            "rho_diff": divergence.get("rho_diff"),
        }

        if identity:
            supporting["identity"] = {
                "status": self._evaluate_identity(identity),
                "hash_valid": identity.get("block_hash_valid", True),
                "chain_continuous": identity.get("chain_continuous", True),
            }

        if structure:
            supporting["structure"] = {
                "status": self._evaluate_structure(structure),
                "dag_coherent": structure.get("dag_coherent", True),
                "cycle_detected": structure.get("cycle_detected", False),
            }

        return supporting

    @staticmethod
    def _mark_contributed(chain: List[AttributionStep], signal: str) -> None:
        """Mark a signal as having contributed to the diagnosis."""
        for step in chain:
            if step.signal == signal:
                step.contributed_to_diagnosis = True
                break


__all__ = [
    "P5DivergenceInterpreter",
    "RootCauseHypothesis",
    "DiagnosticAction",
    "DiagnosticResult",
    "AttributionStep",
    "SupportingSignals",
]
```

### 4.2 Convenience Functions

```python
# # REAL-READY — Convenience functions

def interpret_p5_divergence(
    divergence_snapshot: Dict[str, Any],
    replay_signal: Dict[str, Any],
    topology_signal: Dict[str, Any],
    budget_signal: Dict[str, Any],
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function for P5 divergence interpretation.

    SHADOW MODE: Output is observational only.

    Args:
        divergence_snapshot: P4 divergence snapshot
        replay_signal: Replay safety governance signal
        topology_signal: Topology bundle signal
        budget_signal: Budget signal
        **kwargs: Additional arguments (identity_signal, structure_signal, cycle, run_id)

    Returns:
        Diagnostic panel dict
    """
    interpreter = P5DivergenceInterpreter()
    return interpreter.interpret_divergence(
        divergence_snapshot=divergence_snapshot,
        replay_signal=replay_signal,
        topology_signal=topology_signal,
        budget_signal=budget_signal,
        **kwargs,
    )


def diagnose_from_p4_artifacts(
    divergence_log_path: str,
    replay_signal: Dict[str, Any],
    topology_signal: Dict[str, Any],
    budget_signal: Dict[str, Any],
    cycle_index: int = -1,
) -> Dict[str, Any]:
    """
    Generate diagnostic from existing P4 divergence log.

    SHADOW MODE: Output is observational only.

    Args:
        divergence_log_path: Path to p4_divergence_log.jsonl
        replay_signal: Current replay safety signal
        topology_signal: Current topology signal
        budget_signal: Current budget signal
        cycle_index: Index into divergence log (-1 for latest)

    Returns:
        Diagnostic panel dict
    """
    import json
    from pathlib import Path

    log_path = Path(divergence_log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Divergence log not found: {log_path}")

    # Read JSONL and get specified entry
    entries = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        raise ValueError("Divergence log is empty")

    snapshot = entries[cycle_index]

    return interpret_p5_divergence(
        divergence_snapshot=snapshot,
        replay_signal=replay_signal,
        topology_signal=topology_signal,
        budget_signal=budget_signal,
        cycle=snapshot.get("cycle", 0),
    )
```

---

## 5. Smoke-Test Readiness Checklist

### 5.1 Prerequisites

| Item | Source | Status |
|------|--------|--------|
| P4 divergence_log.jsonl | `results/first_light/p4/*/divergence_log.jsonl` | Required |
| Replay safety signal | `to_governance_signal_for_replay_safety()` | Required |
| Topology bundle signal | `build_topology_bundle_signal()` | Required |
| Budget signal | `build_budget_governance_signal()` | Required |
| Identity signal (optional) | GGFL identity layer | Optional |
| Structure signal (optional) | GGFL structure layer | Optional |

### 5.2 Smoke Test Script

```python
# # REAL-READY — tests/p5/test_divergence_diagnostic_smoke.py

"""
Smoke test for P5 Divergence Diagnostic Panel.

Validates that the interpreter can process P4 artifacts and produce
valid diagnostic output.
"""

import json
from pathlib import Path

import pytest


def test_smoke_nominal_case():
    """Smoke test: All systems nominal produces NOMINAL diagnosis."""
    from backend.analytics.p5_divergence_interpreter import interpret_p5_divergence

    divergence_snapshot = {
        "cycle": 100,
        "severity": "NONE",
        "type": "NONE",
        "divergence_pct": 0.005,
    }

    replay_signal = {
        "status": "OK",
        "governance_alignment": "aligned",
        "conflict": False,
        "reasons": [],
    }

    topology_signal = {
        "mode": "STABLE",
        "persistence_drift": 0.02,
        "within_omega": True,
    }

    budget_signal = {
        "stability_class": "STABLE",
        "health_score": 95,
    }

    result = interpret_p5_divergence(
        divergence_snapshot=divergence_snapshot,
        replay_signal=replay_signal,
        topology_signal=topology_signal,
        budget_signal=budget_signal,
    )

    assert result["root_cause_hypothesis"] == "NOMINAL"
    assert result["action"] == "LOGGED_ONLY"
    assert "schema_version" in result


def test_smoke_structural_break_case():
    """Smoke test: Replay OK + Topology TURBULENT produces STRUCTURAL_BREAK."""
    from backend.analytics.p5_divergence_interpreter import interpret_p5_divergence

    divergence_snapshot = {
        "cycle": 200,
        "severity": "WARN",
        "type": "STATE",
        "divergence_pct": 0.12,
    }

    replay_signal = {
        "status": "OK",
        "governance_alignment": "aligned",
        "conflict": False,
        "reasons": ["[Safety] All checks passed"],
    }

    topology_signal = {
        "mode": "TURBULENT",
        "persistence_drift": 0.18,
    }

    budget_signal = {
        "stability_class": "STABLE",
        "health_score": 90,
    }

    result = interpret_p5_divergence(
        divergence_snapshot=divergence_snapshot,
        replay_signal=replay_signal,
        topology_signal=topology_signal,
        budget_signal=budget_signal,
    )

    assert result["root_cause_hypothesis"] == "STRUCTURAL_BREAK"
    assert result["action"] == "REVIEW_RECOMMENDED"


def test_smoke_replay_failure_case():
    """Smoke test: Replay BLOCK + stable env produces REPLAY_FAILURE."""
    from backend.analytics.p5_divergence_interpreter import interpret_p5_divergence

    divergence_snapshot = {
        "cycle": 300,
        "severity": "WARN",
        "type": "OUTCOME",
        "divergence_pct": 0.08,
    }

    replay_signal = {
        "status": "BLOCK",
        "governance_alignment": "aligned",
        "conflict": False,
        "reasons": ["[Safety] Hash mismatch detected"],
    }

    topology_signal = {
        "mode": "STABLE",
        "persistence_drift": 0.02,
    }

    budget_signal = {
        "stability_class": "STABLE",
        "health_score": 92,
    }

    result = interpret_p5_divergence(
        divergence_snapshot=divergence_snapshot,
        replay_signal=replay_signal,
        topology_signal=topology_signal,
        budget_signal=budget_signal,
    )

    assert result["root_cause_hypothesis"] == "REPLAY_FAILURE"
    assert result["action"] == "INVESTIGATION_REQUIRED"


def test_smoke_json_serializable():
    """Smoke test: Output is JSON serializable."""
    from backend.analytics.p5_divergence_interpreter import interpret_p5_divergence

    result = interpret_p5_divergence(
        divergence_snapshot={"severity": "INFO", "type": "NONE"},
        replay_signal={"status": "OK"},
        topology_signal={"mode": "STABLE"},
        budget_signal={"stability_class": "STABLE"},
    )

    # Should not raise
    json_str = json.dumps(result)
    assert isinstance(json_str, str)

    # Round-trip
    parsed = json.loads(json_str)
    assert parsed["schema_version"] == "1.0.0"


def test_smoke_from_p4_artifacts():
    """Smoke test: Can generate diagnostic from P4 artifacts (if available)."""
    from backend.analytics.p5_divergence_interpreter import diagnose_from_p4_artifacts

    # Check if P4 artifacts exist
    p4_log = Path("results/first_light/evidence_pack_first_light/p4_shadow/divergence_log.jsonl")
    if not p4_log.exists():
        pytest.skip("P4 divergence log not available")

    replay_signal = {"status": "OK", "governance_alignment": "aligned", "conflict": False}
    topology_signal = {"mode": "STABLE", "persistence_drift": 0.02}
    budget_signal = {"stability_class": "STABLE", "health_score": 90}

    result = diagnose_from_p4_artifacts(
        divergence_log_path=str(p4_log),
        replay_signal=replay_signal,
        topology_signal=topology_signal,
        budget_signal=budget_signal,
    )

    assert "root_cause_hypothesis" in result
    assert "action" in result
```

### 5.3 Readiness Criteria

| Criterion | Check | Pass Condition |
|-----------|-------|----------------|
| Schema compliance | Output matches p5_divergence_diagnostic.schema.json | All required fields present |
| JSON serializable | `json.dumps(output)` succeeds | No exceptions |
| Determinism | Same inputs → same outputs | Multiple runs identical |
| Rule coverage | All 8 rules can be triggered | Smoke tests cover NOMINAL, STRUCTURAL_BREAK, REPLAY_FAILURE |
| P4 artifact compatibility | Can read divergence_log.jsonl | `diagnose_from_p4_artifacts()` succeeds |
| SHADOW MODE | Output has no control flow side effects | Observational only |

### 5.4 Integration Path

```
P4 Artifacts                    New P5 Telemetry
     │                                │
     ▼                                ▼
divergence_log.jsonl         Real telemetry signals
     │                                │
     └──────────┬─────────────────────┘
                │
                ▼
      P5DivergenceInterpreter.interpret_divergence()
                │
                ▼
      p5_divergence_diagnostic.json
                │
                ▼
      Evidence Pack / Director Panel / Audit Log
```

---

## 6. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-11 | Initial specification |

---

*Document Status: SPECIFICATION*
*Binding: P5 Implementation*
*SHADOW MODE: All output is observational only*
