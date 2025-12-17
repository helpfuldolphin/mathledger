# Governance Signals as the Nerves of the Organism

## 1. Introduction: The Sentient Organization

In the complex, adaptive system that is our organism, **Governance Signals** are the sensory data, the nerve impulses that allow for self-awareness, self-regulation, and intelligent action. They are the fundamental carriers of information that report on the health, behavior, and integrity of every component, from the lowest-level substrate to the highest-level strategic decision-making.

This document, in the spirit of Phase 0 doctrine, serves as a field manual for understanding these signals. It is not a technical specification but a conceptual guide. It explains what Governance Signals are, why they are essential at every layer of the system, and how they fuse into a singular, coherent "license to operate." This is the core of the "Seatbelt for AI" — a system of active restraints and awareness that ensures the organism remains aligned with its mission and values.

## 2. What is a Governance Signal?

A **Governance Signal** is a structured, machine-readable assertion about the state of a component or process within the organism. It is a declaration that answers a fundamental question: "Is this component behaving as expected, and is it in alignment with its design and purpose?"

Semantically, a Governance Signal is not just raw data. It is **evidence-backed telemetry**. Each signal carries with it a cryptographic signature of its origin, a timestamp, and a clear statement of its meaning. For example, a signal from the Substrate Identity layer might assert: "I am component XYZ, and I attest to my identity with this cryptographic proof."

The key attributes of a Governance Signal are:

*   **Attributable:** It is always clear who or what generated the signal.
*   **Verifiable:** The integrity and authenticity of the signal can be independently verified.
*   **Timely:** The signal is generated and propagated in a timeframe relevant to its purpose.
*   **Meaningful:** The signal has a clear, unambiguous semantic meaning within the governance framework.

## 3. The Symphony of Signals: A Multi-Layered Perspective

The organism is a hierarchical system of systems, each with its own unique functions and responsibilities. To achieve a holistic understanding of the organism's health, we must gather Governance Signals from every layer. Each layer provides a unique and indispensable perspective.

### 3.1. The Vertical Stack of Governance

Imagine the organism as a vertical stack, with each layer building upon the one below it. At each level, a new class of Governance Signals is generated, providing an increasingly abstract and sophisticated view of the system's state.

**(Diagram 1: The Vertical Stack)**

```
      +--------------------------------+
      |      Global Decision           |  (License to Operate)
      +--------------------------------+
                  ^
                  | (Fusion)
      +--------------------------------+
      |       Governance Signals       |
      |--------------------------------|
      | - Replay & Forensics           |
      | - Topology & Bundle Integrity  |
      | - Metrics & Budget Adherence   |
      | - Telemetry & Anomaly Detection|
      | - Substrate & Slice Identity   |
      | - Structure (DAG/HT) Integrity |
      +--------------------------------+
                  ^
                  | (Aggregation)
      +--------------------------------+
      |        System Layers           |
      +--------------------------------+
```

*   **Substrate & Slice Identity:** The foundational layer. Signals from this layer attest to the cryptographic identity and integrity of the underlying hardware and software. *Is this machine the machine it claims to be?*
*   **Structure (DAG/HT):** This layer provides signals about the integrity of the organism's core data structures. *Is the causal history of the organism intact and untampered?*
*   **Telemetry & Anomaly Detection:** The raw feed of operational data. Signals from this layer report on the fine-grained behavior of individual components. *Are the vital signs of this component within normal parameters?*
*   **Metrics & Budget Adherence:** This layer provides signals about resource consumption and performance. *Is this component operating within its allocated budget of time, compute, and other resources?*
*   **Topology & Bundle Integrity:** This layer provides signals about the relationships between components. *Are the components of this bundle the correct versions, and are they communicating in the expected way?*
*   **Replay & Forensics:** The "black box" of the organism. Signals from this layer provide the ability to replay and analyze past events. *Can we reconstruct and understand a past failure or success?*
*   **Adversarial Coverage:** The adversarial testing layer. Signals from this layer report on metric robustness under fault injection, mutation, and replay scenarios. *Are we probing the right places, and do metrics have sufficient failover coverage?*

## 4. Fusion: From Signals to a Singular Decision

The diverse streams of Governance Signals from each layer are not independent. They are fused together in a continuous, real-time process to produce a single, global "license to operate." This is not a simple "on/off" switch but a nuanced spectrum of operational states.

The fusion process is guided by the principles of **defense in depth** and **graceful degradation**. A single anomalous signal from a non-critical component might trigger an advisory mode, raising a flag for human review. A cascade of critical signals, however, could trigger an immediate abort gate, halting a process or isolating a component to prevent further damage.

This is where the evidence from P3/P4 becomes critical. Our understanding of failure modes and the cascading effects of anomalies is encoded in the fusion logic. We have learned that a small deviation in the Substrate Identity layer can be a leading indicator of a catastrophic failure at the Topology layer. The fusion logic is not static; it is a living system that is constantly updated and refined as we gather more data.

## 5. The Nervous System: A Metaphor for Action

If the Governance Signals are the nerves, then the fusion process is the central nervous system, and the resulting actions are the reflexes, decisions, and coordinated movements of the organism.

**(Diagram 2: The Nervous System)**

```
      +---------------------+      +---------------------+      +---------------------+
      |   Governance        |----->|   Fusion Engine     |----->|   Actuators         |
      |   Signals           |      | (Decision Logic)    |      | (Reflexes & Gates)  |
      +---------------------+      +---------------------+      +---------------------+
          |                     ^                     |
          | (Sensory Input)     | (Feedback Loop)     | (Motor Output)
          v                     |                     v
      +-------------------------------------------------------------------------+
      |                           The Organism                                  |
      +-------------------------------------------------------------------------+
```

*   **Reflexes:** These are the fastest, most automatic responses to critical signals. An example is the **abort gate**, which immediately halts a process when a severe anomaly is detected.
*   **Advisory Modes:** These are less drastic responses, designed to provide information and guidance to human operators. A signal that a component is nearing its budget limit might trigger an advisory, suggesting a course of action.
*   **Coordinated Actions:** The fusion engine can also trigger complex, multi-component actions. For example, in response to a detected security threat, it might initiate a process to isolate the affected components, deploy a patch, and trigger a forensic replay to understand the root cause.

## 6. Adversarial Coverage Grid: Council Interpretation

### 6.1 Overview

The **Adversarial Coverage Grid** provides a cross-experiment view of adversarial pressure and failover sufficiency across calibration experiments (CAL-EXP-1, CAL-EXP-2, CAL-EXP-3, etc.). This grid helps answer: "Are we probing the right places, and do metrics have sufficient failover coverage?"

### 6.2 Grid Structure

The grid aggregates per-experiment coverage snapshots into:

- **Pressure Band Counts:** Distribution of experiments across (P3_band, P4_band) combinations
- **Experiments Missing Failover:** List of calibration experiment IDs where `has_failover == False`
- **Experiments by Pressure Band:** Grouping of experiment IDs by pressure band combination

### 6.3 Council Interpretation (Advisory Only)

**Status:** SHADOW MODE — Advisory only, not a hard gate.

**Reading the Grid:**

1. **Pressure Band Distribution:**
   - `(HIGH, HIGH)`: Both P3 and P4 show high adversarial pressure → highest priority for scenario expansion
   - `(MEDIUM, MEDIUM)` or mixed: Moderate pressure → monitor and expand coverage
   - `(LOW, LOW)`: Low pressure → adequate coverage

2. **Missing Failover Detection:**
   - Experiments in `experiments_missing_failover` lack failover coverage
   - If core metrics (`goal_hit`, `density`) are missing failover → council should flag for attention
   - Cross-experiment patterns: if multiple experiments show missing failover → systemic coverage gap

3. **Priority Scenarios:**
   - Top 5 priority scenarios across experiments indicate where new adversarial tests are most needed
   - Council should review these scenarios for implementation priority

**Council Decision Guidance:**

- **BLOCK:** HIGH pressure AND core metrics lack failover (tightened rule)
- **WARN:** MEDIUM pressure OR non-core metrics missing failover
- **OK:** LOW pressure AND all metrics have failover

**Reason-Code Drivers:**

The GGFL adapter uses reason-code drivers to avoid interpretive drift:
- `DRIVER_MISSING_FAILOVER_COUNT`: Indicates experiments missing failover coverage
- `DRIVER_REPEATED_PRIORITY_SCENARIOS`: Indicates priority scenarios repeated across experiments

Drivers are ordered deterministically: missing failover first, then repeated scenarios.

**Note:** This grid is advisory-only. It provides visibility into adversarial coverage gaps but does not enforce promotion gates. The council should use this information to inform decisions about metric promotion readiness, but the grid itself does not block promotions.

### 6.4 Integration Points

- **Per-Experiment Snapshots:** Emitted via `emit_cal_exp_adversarial_coverage()` and persisted to `calibration/adversarial_coverage_<cal_id>.json`
- **Grid Aggregation:** Built via `build_adversarial_coverage_grid()` from snapshot list
- **Evidence Attachment:** Attached to evidence packs under `evidence["governance"]["adversarial_coverage_panel"]` via `attach_adversarial_coverage_grid_to_evidence()`

## 7. Conclusion: The Seatbelt for AI

The Governance Signals framework is the embodiment of the "Seatbelt for AI" philosophy. It is not a system for preventing all failures, but a system for ensuring that when failures do occur, they are detected quickly, their impact is minimized, and the organism can learn and adapt.

By weaving a rich, multi-layered tapestry of signals, we create a system that is not only robust and resilient but also transparent and accountable. Every action, every decision, every heartbeat of the organism is recorded, reported, and available for review. This is the foundation of trust, and the key to unlocking the full potential of this powerful technology.

## 8. Failsafe Invariants: Ensuring System Resilience

To uphold the "Seatbelt for AI" principle, the Governance Engine is designed with critical failsafe invariants. These ensure that even under adverse conditions, the engine's behavior remains predictable, auditable, and non-catastrophic. The primary goal is to prevent cascading failures and provide clear diagnostic information.

### 8.1. Missing Dependency Behavior

The Governance Engine relies on external Python libraries (`pyyaml` for parsing policy files, and `jsonschema` for validating policy structure). While these are declared as hard dependencies, environmental discrepancies can occur.

*   **Behavior:** If either `pyyaml` or `jsonschema` is not found during the engine's initialization, the engine will immediately transition into an **unstable** state. A critical message detailing the missing dependency will be logged.
*   **Impact:** The engine will not attempt to load or process any policy rules.
*   **Recovery:** Normal operation requires the missing dependency to be installed and the engine (or the application hosting it) to be re-initialized.

### 8.2. Invariant `UNSTABLE_POLICY` Semantics

When the Governance Engine is in an unstable state (due to missing dependencies, an invalid policy file, or other internal initialization failures), its `decide()` method will *always* return a predefined `UNSTABLE_POLICY` decision.

*   **Action:** `UNSTABLE_POLICY`
*   **Triggering Rule:** "Engine in failsafe mode due to invalid policy or missing dependencies"

### 8.3. Why Crashing is Forbidden

Crashing is an unacceptable outcome for the Governance Engine. As a critical component responsible for maintaining the "license to operate," an uncontrolled failure could lead to:
*   **Loss of Visibility:** Inability to monitor the health and behavior of other system components.
*   **Unforeseen State:** Potential for other parts of the system to proceed without governance oversight, leading to unconstrained operation.
*   **Cascading Failure:** The engine's crash could trigger other components to fail, exacerbating an already problematic situation.

By failing predictably and entering an `UNSTABLE_POLICY` state, the Governance Engine ensures that the system is aware of the governance failure, can react appropriately (e.g., by logging alerts, halting operations), and remains observable. This allows for controlled degradation rather than catastrophic collapse.
