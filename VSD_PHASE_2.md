# VSD Phase II: Operation Asymmetry

This document outlines the governance and operational parameters for Phase II of the MathLedger project.

## Absolute Safeguards

- **PHASE II Artifacts**: All Phase II artifacts must be clearly labeled “PHASE II — NOT USED IN PHASE I”.
- **Determinism**: All code must remain deterministic except for the random shuffle in the baseline policy.
- **Verifiable Feedback**: RFL uses verifiable feedback only. No RLHF, preferences, or proxy rewards are permitted.
- **No Modification of Phase I**: All new files must be standalone and MUST NOT modify Phase I behavior.
- **Guardrail**: No Phase I log may be used as uplift evidence.

## Phase II Uplift Gate Definition

The Phase II Uplift Gate ensures that all uplift experiments are conducted with rigor, transparency, and determinism.

### 1. Preregistration
All U2 uplift runs must be preregistered in a `PREREG_UPLIFT_U2.yaml` file. This file serves as an immutable record of experimental intent.

### 2. Slice Configuration Hashing
Each experimental slice configuration must be hashed. This hash guarantees the integrity and reproducibility of the slice setup.

### 3. Deterministic Seed Schedule
The experiment must follow a deterministic seed schedule to ensure that results are repeatable and not the product of chance.

### 4. Success Metrics
Uplift will be measured against the following success metrics:
- **`goal_hit`**: Rate of achieving the primary objective.
- **`sparse_density`**: Measure of the efficiency of the solution path.
- **`chain_success`**: Success rate across a chain of dependent tasks.
- **`joint_goal`**: Rate of achieving a composite goal of multiple objectives.

### 5. CI / Wilson CI Requirements
All experiments must pass Continuous Integration (CI) checks, including Wilson CI for statistical significance, before results can be considered valid.

## Escalation Path
In the event that two or more slices demonstrate statistically significant uplift, the following escalation path will be triggered:
1.  **Immediate Halt**: All active uplift experiments will be paused.
2.  **Audit**: A full audit of the uplift evidence will be conducted by the Basis Governance team.
3.  **Report**: A report will be delivered to stakeholders detailing the findings.
4.  **Decision**: A decision on whether to proceed with Phase III will be made based on the audit report.
