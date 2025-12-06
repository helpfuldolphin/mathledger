# Canonical Basis Plan: Phase II

This document outlines the plan for managing the canonical basis during Phase II.

## Absolute Safeguards

- **PHASE II Artifacts**: All Phase II artifacts must be clearly labeled “PHASE II — NOT USED IN PHASE I”.
- **Determinism**: All code must remain deterministic except for the random shuffle in the baseline policy.
- **Verifiable Feedback**: RFL uses verifiable feedback only. No RLHF, preferences, or proxy rewards are permitted.
- **No Modification of Phase I**: All new files must be standalone and MUST NOT modify Phase I behavior.
- **Guardrail**: No Phase I log may be used as uplift evidence.

## Phase II Uplift Gate Definition

The Phase II Uplift Gate is a critical control for ensuring the integrity of the canonical basis.

### 1. Preregistration
Uplift experiments (U2 runs) must be preregistered in `PREREG_UPLIFT_U2.yaml`. This ensures that any changes to the basis are preceded by a clear statement of intent.

### 2. Slice Configuration Hashing
Slice configurations are hashed to prevent unauthorized or accidental modifications to the experimental setup.

### 3. Deterministic Seed Schedule
A deterministic seed schedule is enforced to ensure that any observed uplift is a direct result of the experimental changes, not random variation.

### 4. Success Metrics
The success of an uplift experiment will be evaluated based on a predefined set of metrics:
- **`goal_hit`**
- **`sparse_density`**
- **`chain_success`**
- **`joint_goal`**

### 5. CI / Wilson CI Requirements
All changes must pass CI and Wilson CI checks to confirm that they do not destabilize the system and that any observed uplift is statistically significant.

## Escalation Path
If two or more slices show significant uplift, the following protocol is enacted:
1.  **Pause Experiments**: All ongoing uplift experiments are halted.
2.  **Comprehensive Audit**: The Basis Governance team initiates a comprehensive audit of the results.
3.  **Stakeholder Report**: A detailed report is prepared and distributed to all relevant stakeholders.
4.  **Governance Decision**: A formal governance decision will be made regarding the integration of the successful changes into the canonical basis.
