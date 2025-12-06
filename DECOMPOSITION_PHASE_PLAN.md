# MathLedger Phase II Decomposition Plan

This document outlines the Work Breakdown Structure (WBS) for Phase II of the MathLedger project, focusing on the uplift experiments.

## PHASE II — UPLIFT WORK BREAKDOWN STRUCTURE

**NOTICE:** PHASE II — NOT RUN IN PHASE I

Complete work breakdown structure for Phase II uplift experiments implementing asymmetric environments, U2 runner, slice-specific success metrics, and evidence pack integration.

**Governance Gate:** `VSD_PHASE_2_uplift_gate`

**Preregistration Document:** `experiments/prereg/PREREG_UPLIFT_U2.yaml`

**Curriculum Document:** `config/curriculum_uplift_phase2.yaml`

### Ground Rules
- **No Phase I Reinterpretation:** Yes
- **Phase Ii Artifacts Labeled:** Yes
- **Strict Preregistration Adherence:** Yes
- **Rfl Verifiable Feedback Only:** Yes
- **Determinism Required:** Yes

### Task Summary
- **UPLIFT-100:** Implement Phase II curriculum loader
- **UPLIFT-110:** Implement slice success metric evaluators
- **UPLIFT-120:** Implement run_uplift_u2.py
- **UPLIFT-130:** Implement slice diagnostics + chain-depth analyzer
- **UPLIFT-140:** Implement summary + CI harness
- **UPLIFT-150:** Evidence Pack v2 integration

---

### Detailed Task Descriptions

#### UPLIFT-100: Implement Phase II curriculum loader

**Description:** Create CurriculumLoaderV2 that reads curriculum_uplift_phase2.yaml and provides slice configurations for the four asymmetric environments (goal, sparse, tree, dependency). Must validate slice parameters against schema and expose slice_params, success_metric, and budget constraints.

**Properties:**
- `requires_phase`: "II"
- `blocked_by`: ["uplift_gate"]
- `not_allowed_in_phase_I`: true

**Acceptance Criteria:**
- Loads all 4 Phase II slices from curriculum_uplift_phase2.yaml
- Validates required fields: atoms, depth_min, depth_max, breadth_max, total_max
- Exposes success_metric configuration per slice
- Returns SliceConfig dataclass with all parameters
- Unit tests pass for each slice type
- Error handling for malformed YAML

---

#### UPLIFT-110: Implement goal_hit success metric evaluator

**Description:** Create SuccessEvaluatorGoalHit that determines cycle success based on whether target formula hashes were proven. Implements the goal_hit metric from PREREG_UPLIFT_U2.yaml with min_goal_hits and min_total_verified thresholds.

**Properties:**
- `requires_phase`: "II"
- `blocked_by`: ["uplift_gate"]
- `not_allowed_in_phase_I`: true

**Acceptance Criteria:**
- evaluate(cycle_results, config) returns SuccessResult
- SuccessResult contains: success (bool), goal_hits (int), total_verified (int), details (dict)
- Matches target hashes from success_metric.parameters.target_hashes
- Returns success=True iff goal_hits >= min_goal_hits AND total_verified >= min_total_verified
- Unit tests with mock cycle results

---

#### UPLIFT-120: Implement run_uplift_u2.py core runner

**Description:** Create the U2 experiment runner that executes paired baseline/RFL runs on all four Phase II slices. Uses CurriculumLoaderV2 for slice configs and SuccessEvaluatorFactory for per-cycle success evaluation. Follows PREREG_UPLIFT_U2.yaml protocol.

**Properties:**
- `requires_phase`: "II"
- `blocked_by`: ["uplift_gate"]
- `not_allowed_in_phase_I`: true

**Acceptance Criteria:**
- CLI interface: --slice, --cycles, --seed, --output-dir
- Runs baseline mode with random_shuffle candidate ordering
- Runs RFL mode with policy_score candidate ordering
- Uses deterministic seeding: MDAP_SEED + cycle_index
- Logs cycle-level success metrics to JSONL
- Generates experiment_manifest.json matching PREREG schema
- Handles all 4 slice types correctly

---

#### UPLIFT-130: Implement chain-depth analyzer for tree slice diagnostics

**Description:** Create ChainDepthAnalyzer that computes derivation chain statistics: max_depth, mean_depth, depth_histogram, deepest_formulas. Used for slice_uplift_tree success evaluation and post-hoc analysis.

**Properties:**
- `requires_phase`: "II"
- `blocked_by`: ["uplift_gate"]
- `not_allowed_in_phase_I`: true

**Acceptance Criteria:**
- analyze(proof_dag, verified_hashes) returns ChainDepthStats
- ChainDepthStats contains: max_depth, mean_depth, depth_histogram, deepest_n
- Efficient BFS/DFS traversal for large DAGs
- Handles cyclic DAG gracefully (error, not infinite loop)
- Visualization helper for depth distribution

---

#### UPLIFT-140: Implement U2 statistical summary generator

**Description:** Create StatisticalSummaryU2 that computes primary and secondary metrics per PREREG_UPLIFT_U2.yaml: success_rate with Wilson CI, Δp with bootstrap CI, two-proportion z-test or bootstrap comparison. Outputs statistical_summary.json.

**Properties:**
- `requires_phase`: "II"
- `blocked_by`: ["uplift_gate"]
- `not_allowed_in_phase_I`: true

**Acceptance Criteria:**
- generate_summary(baseline_log, rfl_log, config) returns StatisticalSummary
- Primary metric: success_rate per mode with 95% Wilson CI
- Computes Δp = p_rfl - p_base with bootstrap CI
- Two-proportion z-test p-value
- Secondary metrics: verified_per_cycle, efficiency
- Effect size Cohen's h computed
- JSON output matches preregistered schema

---

#### UPLIFT-150: Design Evidence Pack v2 structure

**Description:** Design directory structure and manifest schema for Evidence Pack v2 containing Phase II uplift results. Define attestation format, artifact checksums, and cross-references to Phase I pack.

**Properties:**
- `requires_phase`: "II"
- `blocked_by`: ["uplift_gate"]
- `not_allowed_in_phase_I`: true

**Acceptance Criteria:**
- docs/evidence/EVIDENCE_PACK_V2_STRUCTURE.md written
- Schema includes: manifest.json, per-slice results, statistical_summary.json
- Cross-reference to Evidence Pack v1 without modification
- Attestation format for Phase II results
- Artifact checksums for all files

---
