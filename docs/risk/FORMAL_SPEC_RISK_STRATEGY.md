# Formal Specification for Risk-Informed Strategy

This document provides the formal specification for the risk-informed strategy in MathLedger, including data schemas, metric thresholds, and the aggregation methodology.

## 1. JSON Schema: Risk Assessment Tile

This JSON schema defines the structure for a "Risk Assessment Tile," a standardized data object for reporting and evaluating individual risk metrics. The schema conforms to JSON Schema Draft-07.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Risk Assessment Tile",
  "description": "A structured representation of a single risk metric assessment for a MathLedger component, providing a snapshot of performance against a defined threshold.",
  "type": "object",
  "properties": {
    "metric_id": {
      "description": "A unique, machine-readable identifier for the metric.",
      "type": "string",
      "examples": ["delta_p", "rsi", "omega", "tda_divergence"]
    },
    "metric_name": {
      "description": "A human-readable name for the metric.",
      "type": "string",
      "examples": ["Performance Delta", "Robustness Stress Index", "Generalization Score (Omega)", "TDA Divergence"]
    },
    "value": {
      "description": "The measured value of the metric.",
      "type": "number"
    },
    "threshold": {
      "description": "The specific gating threshold for this metric. The interpretation depends on the 'operator' field.",
      "type": "number"
    },
    "operator": {
      "description": "The mathematical operator used for the gate check (e.g., value > threshold).",
      "type": "string",
      "enum": [">", ">=", "<", "<=", "=="]
    },
    "gate": {
      "description": "The governance gate this metric influences.",
      "type": "string",
      "enum": ["P3", "P4", "Deployment", "Defense Compliance"]
    },
    "risk_band": {
      "description": "The calculated risk level based on the metric's value relative to its threshold.",
      "type": "string",
      "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    },
    "timestamp": {
      "description": "The ISO 8601 timestamp of when the metric was measured.",
      "type": "string",
      "format": "date-time"
    },
    "justification_ref": {
      "description": "A reference or link to the documentation justifying the threshold value.",
      "type": "string",
      "format": "uri-reference"
    }
  },
  "required": [
    "metric_id",
    "metric_name",
    "value",
    "threshold",
    "operator",
    "gate",
    "risk_band",
    "timestamp"
  ]
}
```

## 2. Gating Thresholds

The following are the exact, authoritative gating thresholds for the primary risk metrics. These values are illustrative and would be derived from rigorous testing and safety analysis in a real system.

| Metric | Gate | Operator | Threshold | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Δp** (Perf. Delta) | P3 | `<` | `0.05` | Performance degradation on core curriculum must be less than 5%. |
| **RSI** (Robustness) | P3 | `>` | `0.80` | Must successfully pass at least 80% of the adversarial robustness stress tests. |
| **Ω** (Generalization) | P3 | `>` | `0.95` | Must maintain at least 95% performance on the out-of-distribution generalization suite. |
| **TDA** (Domain Aware.)| P3 | `>` | `0.99` | Must correctly identify >99% of known out-of-distribution inputs. |
| **Divergence** | P4 | `<` | `0.02` | Mission-specific performance must not diverge more than 2% from human-in-the-loop baseline. |

## 3. Risk Aggregation Formula

To calculate an overall `risk_band`, individual metric scores are normalized and combined using a weighted average.

1.  **Normalization**: Each metric's `value` is normalized to a risk score `S_i` between 0 (no risk) and 1 (maximum risk) based on its distance from the threshold. For a "greater than" operator, the formula is: `S_i = max(0, 1 - (value / threshold))`. For a "less than" operator: `S_i = max(0, (value / threshold) - 1)`.

2.  **Weighted Sum**: The overall risk score is the weighted sum of individual scores:
    `OverallRisk = Σ(w_i * S_i)`

    *   **Weights (`w_i`)**:
        *   `w_Δp`: 0.15
        *   `w_RSI`: 0.30
        *   `w_Ω`: 0.25
        *   `w_TDA`: 0.10
        *   `w_Divergence`: 0.20

3.  **Banding**: The `OverallRisk` score maps to a qualitative `risk_band`:

| OverallRisk Score | Risk Band |
| :--- | :--- |
| `< 0.10` | **LOW** |
| `0.10 - 0.39` | **MEDIUM** |
| `0.40 - 0.69` | **HIGH** |
| `>= 0.70` | **CRITICAL** |

## 4. Regulator-Facing Justification

**Subject: Justification of Quantitative Thresholds for P3/P4 Governance Gates**

This document provides the rationale for the specific, quantitative gating thresholds established within the MathLedger system, directly linking them to our P3 (Pre-Training) and P4 (Post-Training) governance doctrines. Our approach is rooted in quantitative control theory, not subjective policy, to ensure system safety, predictability, and regulatory compliance.

**P3 Doctrine: Foundational Capability & Intrinsic Safety**

The P3 gate certifies a model's foundational capabilities and intrinsic safety *before* it is considered for mission-specific adaptation. The metrics and their stringent thresholds are designed to be context-independent and serve as a non-negotiable quality floor.

*   **Thresholds for Δp (<0.05), RSI (>0.80), and Ω (>0.95)** are derived from large-scale statistical analysis of historical model runs. They represent the 99th percentile of performance for models that have demonstrated stable and reliable behavior. This ensures that any model passing the P3 gate possesses a verified high degree of performance, robustness against known failure modes, and the ability to generalize, which is critical for preventing catastrophic failures when faced with novel inputs.

**P4 Doctrine: Mission Adaptation & Operational Safety**

The P4 gate evaluates a model's fitness for a *specific operational context*. Its metrics are designed to detect subtle but critical shifts in behavior that could compromise safety or mission objectives in a live environment.

*   **The Divergence threshold (<0.02)** is directly tied to human-in-the-loop (HITL) performance baselines. This threshold represents the maximum tolerable deviation from verified human expert judgment in a given task. It is a critical control to prevent autonomous systems from "drifting" into unsafe or undesirable operational regimes that were not explicitly authorized. This threshold is derived from safety-case analyses and is a direct implementation of our commitment to maintaining meaningful human control.

By enforcing these explicit, data-driven thresholds at both the P3 and P4 gates, we ensure a continuous, auditable chain of evidence that demonstrates the system's adherence to its safety and performance requirements, from initial training to operational deployment.
