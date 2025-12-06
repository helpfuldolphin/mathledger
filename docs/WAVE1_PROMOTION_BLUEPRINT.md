# WAVE1_PROMOTION_BLUEPRINT.md

---

## ⚠️ STATUS: BLUEPRINT ONLY — BASIS PROMOTION NOT EXECUTED ⚠️

**This document is a PLANNING ARTIFACT for future work.**

**No promotion steps have been executed. No modules have been migrated.**

The `basis/` directory contains pre-existing modules that were developed in parallel
with the spanning set, but the formal promotion process described in this document
has NOT been started.

**Phase Classification**: This entire document describes **Phase II (Future Work)**.
It is NOT part of Evidence Pack v1.

**The existence of additional RFL logs (50-cycle, 330-cycle) does not begin or influence basis promotion.**

---

**Auditor**: Claude O
**Date**: 2025-11-30
**Status**: BLUEPRINT ONLY — NOT EXECUTED

---

## Promotion Execution Status

| Step | Description | Status |
|------|-------------|--------|
| D0.1 | Hash Consolidation Audit | **Not Started** |
| D0.2 | Canonical Normalize Unification Test | **Not Started** |
| D0.3 | Dual-Root Invariant Confirmation | **Not Started** |
| D0.4 | Domain Tag Audit | **Not Started** |
| D0.5 | Deprecation Notices Installation | **Not Started** |
| D1.1 | Verify Existing Basis Tests | **Not Started** |
| D1.2 | Install Import Shims | **Not Started** |
| D1.3 | Update pyproject.toml | **Not Started** |
| D1.4 | Integration Smoke Test | **Not Started** |
| D1.5 | Determinism Verification | **Not Started** |
| D2.1 | Consolidate Normalizers | **Not Started** |
| D2.2 | Consolidate Attestation | **Not Started** |
| D2.3 | Extract RFC 8785 | **Not Started** |
| D2.4 | Extract Determinism Helpers | **Not Started** |
| D2.5 | Full Test Suite Run | **Not Started** |
| D3.1 | RFL Bootstrap Stats Uplift | **Not Started** |
| D3.2 | RFL Config Promotion | **Not Started** |
| D3.3 | Axiom Engine Pure Subset Extraction | **Not Started** |
| D3.4 | Import Graph Validation | **Not Started** |
| D3.5 | Purity Contract Audit | **Not Started** |
| D3.6 | Tag Release v0.2.0-wave1 | **Not Started** |
| D3.7 | Documentation Update | **Not Started** |
| U2.1 | U2 Preregistration Compliance (G1) | Not Started / Passed / Failed |
| U2.2 | U2 Determinism Compliance (G2) | Not Started / Passed / Failed |
| U2.3 | U2 Manifest Compliance (G3) | Not Started / Passed / Failed |
| U2.4 | U2 RFL Integrity Compliance (G4) | Not Started / Passed / Failed |
| U2.5 | U2 Uplift Curve Analysis (non-blocking) | Not Started / Positive / Mixed / Neutral / Negative |
| — | RFL evidence (50-cycle, 330-cycle, 1000-cycle logs) → does not trigger promotion criteria | **N/A** |

**Summary**: 0 of 22 steps completed. Promotion has not begun.

**Note**: RFL experimental evidence is independent of basis promotion. The existence of
RFL logs, Dyno Charts, or attestation artifacts does not satisfy any promotion criterion.

---

## Phase II Uplift as a Non-Blocking Signal (Design Only)

> **⚠️ THIS SECTION IS THEORETICAL DESIGN FOR FUTURE WORK ⚠️**
>
> Nothing described below is implemented. No RFL evidence currently affects promotion.
> This section exists solely to document how a future governance model *might* incorporate
> uplift signals — if and when all prerequisite conditions are satisfied.

### Current State (Binding)

As of this document's date, the following holds:

| Item | Status |
|------|--------|
| RFL 50-cycle logs | **No effect on promotion** |
| RFL 330-cycle logs | **No effect on promotion** |
| RFL 1000-cycle logs | **No effect on promotion** |
| Dyno Charts (baseline vs RFL) | **No effect on promotion** |
| Attestation artifacts (H_t) | **No effect on promotion** |
| Any future RFL logs | **No effect on promotion until conditions below are met** |

### Theoretical Future Model (Design Only — Not Active)

In a future phase, RFL uplift evidence *might* serve as a **non-blocking confidence signal**
for promotion decisions. This would NOT replace the 22-step promotion criteria, but could
provide supplementary confidence that promoted modules behave correctly under load.

**Role of Uplift in Future Governance:**
- **NOT a gating criterion** — Promotion never requires positive uplift
- **NOT a substitute for purity checks** — All 18 purity contract rules remain mandatory
- **Potential role: sanity check** — If promoted modules cause uplift regression, investigate
- **Potential role: confidence bump** — Consistent uplift across slices increases confidence

### Minimum Conditions Before Uplift Could Influence Promotion (All Must Be Satisfied)

| Condition | Description | Current Status |
|-----------|-------------|----------------|
| **C1** | Non-degenerate slice verified | **Not Satisfied** — No slice validation protocol exists |
| **C2** | Reproducible uplift (≥3 independent runs, CI lower bound > 1.0) | **Not Satisfied** — No multi-run uplift verification |
| **C3** | Clean manifests (all artifacts sealed, no gaps) | **Not Satisfied** — Manifest sealing not audited |
| **C4** | ΔH attestation checks pass (H_t stable across runs) | **Not Satisfied** — No ΔH stability verification |
| **C5** | Governance gate approval (human sign-off) | **Not Satisfied** — No governance gate defined |
| **C6** | Baseline established on pre-promotion codebase | **Not Satisfied** — No pre/post comparison framework |
| **C7** | Uplift measured on promoted vs non-promoted paths | **Not Satisfied** — No A/B infrastructure |

**Summary of Conditions**: 0 of 7 satisfied. Uplift has no influence on promotion.

### If All Conditions Were Satisfied (Hypothetical)

If all seven conditions above were satisfied in a future phase, a new row might be added
to the Promotion Execution Status table:

```
| UPL.1 | Uplift sanity check (non-blocking) | Passed/Failed/Skipped |
```

This row would be:
- **Non-blocking** — A "Failed" result would trigger investigation, not block promotion
- **Optional** — Could be marked "Skipped" if uplift infrastructure unavailable
- **Supplementary** — Provides confidence, not correctness guarantees

### Explicit Separation from Current Criteria

The current Promotion Execution Status table row:

```
| — | RFL evidence (50-cycle, 330-cycle, 1000-cycle logs) → does not trigger promotion criteria | N/A |
```

**Remains unchanged and binding.** The theoretical future model described above does not
modify this row. Any future activation of uplift-as-signal would require:

1. Satisfaction of all 7 conditions (C1–C7)
2. Explicit amendment to this document
3. Addition of new row(s) to the Promotion Execution Status table
4. Human governance approval

Until then, **RFL evidence has zero effect on basis promotion**.

---

## PHASE II — NOT RUN IN PHASE I: U2 Uplift Signals as Non-Blocking Evidence

> **⚠️ PHASE II ARTIFACT ⚠️**
>
> This section documents how Phase II U2 (asymmetric uplift environment) evidence
> may serve as **non-blocking supplementary evidence** for promotion decisions.
> All content below is PHASE II design — nothing has been executed in Phase I.

### Overview: U2 Uplift in Governance

Phase II introduces **U2 runners** operating across **four asymmetric uplift environments**.
These environments produce **slice-specific success metrics** that can inform (but never gate)
promotion decisions.

**Key Principle**: U2 uplift signals are **non-blocking evidence** — they provide supplementary
confidence signals but NEVER serve as promotion gates. Promotion remains governed by the
22-step criteria and 18-rule purity contract defined in this document.

### Phase II Non-Blocking Evidence Classification

The following evidence types from Phase II U2 experiments are classified as **non-blocking**:

| Evidence Type | Source | Classification | Usage |
|---------------|--------|----------------|-------|
| **U2 Uplift Curves** | U2 runner output per environment | Non-blocking | Pattern analysis for curriculum tuning |
| **Slice-Specific Metrics** | PREREG_UPLIFT_U2.yaml success metrics | Non-blocking | Regression detection, anomaly flagging |
| **Cross-Environment Ratios** | Aggregate over 4 asymmetric environments | Non-blocking | Confidence scoring |
| **H_t Stability Traces** | Attestation logs across U2 runs | Non-blocking | Determinism verification audit |
| **Coverage Delta (ΔC)** | RFL coverage before/after policy update | Non-blocking | Curriculum advancement signal |
| **Telemetry Aggregates** | Per-cycle metrics from U2 runner | Non-blocking | Operational health monitoring |

**Binding Constraint**: None of the above evidence types may block promotion. They inform
but never gate. The sole blockers remain the 22 promotion steps (D0–D3) and 18 purity rules (A–D).

### Phase II Uplift Signal Categories

| Signal Category | Description | Governance Role |
|-----------------|-------------|-----------------|
| **U2-CURVE** | Slice-specific uplift curves (coverage vs cycles) | Non-blocking: pattern analysis |
| **U2-MAGNITUDE** | Numeric uplift ratios per environment | Non-blocking: regression detection |
| **U2-STABILITY** | Cross-run consistency of uplift patterns | Non-blocking: reproducibility check |
| **U2-ATTESTATION** | H_t stability across U2 runs | Non-blocking: determinism verification |

### Conditions Under Which Uplift Curves Influence Promotion

Uplift curves MAY influence promotion decisions **only when ALL of the following are true**:

#### Tier 1: Prerequisite Conditions (Must Be Met First)

| Condition | Description | Verification Method |
|-----------|-------------|---------------------|
| **P1** | All 7 original conditions (C1–C7) satisfied | See table above |
| **P2** | PREREG_UPLIFT_U2.yaml preregistration in effect | Verify file exists and is sealed |
| **P3** | U2 runner determinism verified (except baseline random) | Compare 2+ runs, verify identical outputs |
| **P4** | Four asymmetric environments properly configured | Environment manifest audit |

#### Tier 2: Evidence Quality Conditions

| Condition | Description | Verification Method |
|-----------|-------------|---------------------|
| **E1** | Uplift measured across ≥3 independent U2 runs | Run log count verification |
| **E2** | Slice-specific success metrics defined per PREREG | Metric registry check |
| **E3** | No manual intervention in any U2 run | Audit log inspection |
| **E4** | Verifiable feedback only (no human preferences, no proxies) | RFL integrity check |

#### Tier 3: Curve Interpretation Criteria

| Criterion | Description | Pass Condition |
|-----------|-------------|----------------|
| **I1** | Uplift curve monotonicity | Non-decreasing coverage over cycles |
| **I2** | Cross-environment consistency | Similar curve shapes across 4 environments |
| **I3** | Statistical significance | Bootstrap CI lower bound > 1.0 for ≥2 environments |
| **I4** | No anomalous regression | No environment shows >10% drop from baseline |

### How Uplift Curves May Influence Promotion (When Conditions Met)

When ALL Tier 1–3 conditions are satisfied, uplift evidence may influence promotion as follows:

| Scenario | Uplift Signal | Effect on Promotion |
|----------|---------------|---------------------|
| **Positive uplift, all environments** | CI > 1.0 everywhere | **Non-blocking confidence boost** — proceed with higher confidence |
| **Mixed uplift** | Some environments show CI > 1.0, others ≤ 1.0 | **Non-blocking, requires investigation** — document before proceeding |
| **Neutral uplift** | CI spans 1.0 in all environments | **No effect** — neither supports nor opposes promotion |
| **Negative uplift** | CI < 1.0 in ≥2 environments | **Non-blocking but triggers review** — investigate root cause before proceeding |
| **Catastrophic regression** | All environments show CI < 0.8 | **Non-blocking but STRONG recommendation to pause** — human review required |

**Critical**: Even in the "catastrophic regression" case, uplift does NOT block promotion.
It triggers a strong recommendation for human review, but the 22-step promotion criteria
remain the sole determinant of promotion eligibility.

### Detailed Uplift Curve Influence Conditions

The following matrix defines precisely when and how uplift curves may influence promotion:

#### Condition Matrix: Curve Shape Analysis

| Curve Pattern | Interpretation | Influence on Promotion |
|---------------|----------------|------------------------|
| **Monotonic increasing** | RFL learning effective | Positive signal — increases confidence |
| **Plateau after ramp** | Curriculum ceiling reached | Neutral — suggests slice advancement needed |
| **Oscillating** | Policy instability or noise | Investigation signal — check seeds, check determinism |
| **Monotonic decreasing** | Regression or degenerate slice | Warning signal — investigate before proceeding |
| **Flat at zero** | No learning detected | Negative control confirmation or degenerate environment |
| **Step function** | Discrete improvement events | Analyze event correlation with policy updates |

#### Condition Matrix: Cross-Environment Consistency

| Consistency Pattern | Interpretation | Influence on Promotion |
|---------------------|----------------|------------------------|
| **All 4 environments positive** | Strong uplift signal | High confidence — proceed with documentation |
| **3 of 4 positive, 1 neutral** | Likely valid uplift | Moderate confidence — note outlier |
| **2 positive, 2 neutral** | Inconclusive | Low confidence — requires investigation |
| **Mixed positive/negative** | Environment asymmetry detected | Pause for analysis — document before proceeding |
| **All 4 neutral** | No detectable uplift | May proceed — no evidence against promotion |
| **All 4 negative** | Strong regression signal | STRONG recommendation to investigate — do not block |

#### Temporal Stability Requirements

For uplift curves to be considered as evidence, the following temporal conditions must hold:

| Temporal Check | Requirement | Verification Method |
|----------------|-------------|---------------------|
| **Run-to-run stability** | ≥3 independent runs with consistent curve shape | Compare curve plots visually + compute variance |
| **Attestation chain continuity** | No H_t gaps in run logs | Validate all cycles have recorded H_t |
| **Policy update traceability** | Each policy delta linked to parent cycle | Check proof_parents linkage in manifest |
| **Seed determinism** | Same seed → identical curve (except baseline random) | Hash comparison of outputs |

### U2 Evidence Compliance Gates

To ensure U2 evidence is valid and trustworthy, the following compliance gates MUST pass
before U2 signals may be considered as non-blocking evidence:

#### Gate G1: Preregistration Compliance

| Check | Description | Pass Condition |
|-------|-------------|----------------|
| **G1.1** | PREREG_UPLIFT_U2.yaml exists | File present in `docs/prereg/` |
| **G1.2** | Preregistration sealed before U2 runs | Seal timestamp < first U2 run timestamp |
| **G1.3** | No amendments after first run | File unchanged post-seal (hash verified) |
| **G1.4** | Hypothesis clearly stated | Preregistration contains testable prediction |

#### Gate G2: Determinism Compliance

| Check | Description | Pass Condition |
|-------|-------------|----------------|
| **G2.1** | U2 runner uses fixed seeds | Config file specifies deterministic seeds |
| **G2.2** | Baseline random policy documented | Random policy is sole exception to determinism |
| **G2.3** | Repeated runs produce identical results | Hash of outputs matches across runs |
| **G2.4** | Attestation roots stable | H_t identical for same inputs |

#### Gate G3: Manifest Compliance

| Check | Description | Pass Condition |
|-------|-------------|----------------|
| **G3.1** | All U2 artifacts listed in manifest | No orphan files in U2 output directories |
| **G3.2** | Manifest sealed after each run | Manifest hash recorded in run log |
| **G3.3** | No gaps in artifact sequence | Consecutive run IDs, no missing outputs |
| **G3.4** | Artifact integrity verified | All artifacts pass hash verification |

#### Gate G4: RFL Integrity Compliance

| Check | Description | Pass Condition |
|-------|-------------|----------------|
| **G4.1** | Verifiable feedback only | No human preference data in feedback loop |
| **G4.2** | No proxy metrics | Direct statement verification, not approximations |
| **G4.3** | Curriculum bounds respected | All derivations within slice bounds |
| **G4.4** | Coverage metrics computed correctly | Audit trail from statements to coverage |

### Compliance Gate Summary

| Gate | Name | Checks | Status |
|------|------|--------|--------|
| **G1** | Preregistration Compliance | G1.1–G1.4 | **Not Evaluated** — No U2 runs |
| **G2** | Determinism Compliance | G2.1–G2.4 | **Not Evaluated** — No U2 runs |
| **G3** | Manifest Compliance | G3.1–G3.4 | **Not Evaluated** — No U2 runs |
| **G4** | RFL Integrity Compliance | G4.1–G4.4 | **Not Evaluated** — No U2 runs |

**Summary**: 0 of 4 gates evaluated. U2 evidence has no current influence on promotion.

### Gate Enforcement Protocol

#### Gate Execution Sequence

Gates MUST be evaluated in order. Failure at any gate halts evaluation:

```
G1 (Preregistration) → G2 (Determinism) → G3 (Manifest) → G4 (RFL Integrity)
     ↓ PASS              ↓ PASS              ↓ PASS            ↓ PASS
   Continue            Continue            Continue         → U2.5 Evaluation
     ↓ FAIL              ↓ FAIL              ↓ FAIL            ↓ FAIL
   HALT + LOG          HALT + LOG          HALT + LOG       HALT + LOG
```

#### Gate Failure Consequences

| Gate Failed | Consequence | Recovery Path |
|-------------|-------------|---------------|
| **G1** | U2 evidence INVALID | Seal PREREG_UPLIFT_U2.yaml, re-run from scratch |
| **G2** | U2 evidence INVALID | Fix nondeterminism source, re-run all experiments |
| **G3** | U2 evidence INVALID | Regenerate manifests, verify artifact chain |
| **G4** | U2 evidence INVALID | Audit RFL feedback loop, remove proxies, re-run |

#### Gate Pass Logging Requirements

When all gates pass, the following must be recorded:

```yaml
# U2_GATE_LOG_ENTRY (append to ops/logs/u2_compliance.jsonl)
gate_evaluation:
  timestamp: "<ISO8601>"
  evaluator: "<agent_id>"
  prereg_hash: "<sha256 of PREREG_UPLIFT_U2.yaml>"
  gates:
    G1: { status: "PASS", checks: [G1.1, G1.2, G1.3, G1.4], evidence: "<path>" }
    G2: { status: "PASS", checks: [G2.1, G2.2, G2.3, G2.4], evidence: "<path>" }
    G3: { status: "PASS", checks: [G3.1, G3.2, G3.3, G3.4], evidence: "<path>" }
    G4: { status: "PASS", checks: [G4.1, G4.2, G4.3, G4.4], evidence: "<path>" }
  u2_5_eligible: true
  promotion_influence: "non-blocking"
```

### U2 Evidence and the Four Asymmetric Environments

Phase II defines four asymmetric uplift environments (per PREREG_UPLIFT_U2.yaml):

| Environment | Asymmetry Type | Success Metric Focus | Slice Bounds |
|-------------|----------------|----------------------|--------------|
| **ENV_A** | Sparse goal structure | `goal_hit`, `sparse_density` | atoms≤4, depth≤4 |
| **ENV_B** | Chain derivation focus | `chain_success`, `joint_goal` | atoms≤4, depth≤5 |
| **ENV_C** | Breadth-constrained | `coverage_ratio`, `novel_count` | atoms≤5, depth≤4 |
| **ENV_D** | Full complexity | All metrics aggregated | atoms≤5, depth≤6 |

Each environment MUST:
1. Be independently seeded (per PREREG)
2. Produce separate uplift curves
3. Have its own manifest sealed after each run
4. Contribute to cross-environment consistency analysis

### Telemetry Requirements for U2 Evidence

U2 runners MUST emit telemetry meeting the following schema:

| Telemetry Field | Type | Required | Description |
|-----------------|------|----------|-------------|
| `cycle_id` | int | Yes | Monotonic cycle counter |
| `environment_id` | string | Yes | ENV_A, ENV_B, ENV_C, or ENV_D |
| `timestamp_monotonic` | int | Yes | Monotonic nanosecond counter (NOT wall clock) |
| `coverage_snapshot` | float | Yes | Current slice coverage [0.0, 1.0] |
| `policy_hash` | hex64 | Yes | SHA-256 of current policy weights |
| `H_t` | hex64 | Yes | Composite attestation root |
| `statements_derived` | int | Yes | Count of new statements this cycle |
| `uplift_ratio` | float | No | Computed ratio vs baseline (if available) |

**Determinism Note**: `timestamp_monotonic` uses a process-local counter, NOT `time.time()`.
This ensures reproducibility while maintaining temporal ordering for analysis.

### Integration with Promotion Execution Status

If and when all compliance gates pass and Tier 1–3 conditions are satisfied, the following
rows will be added to the Promotion Execution Status table (Section above):

```
| U2.1 | U2 Preregistration Compliance (G1) | Not Started / Passed / Failed |
| U2.2 | U2 Determinism Compliance (G2) | Not Started / Passed / Failed |
| U2.3 | U2 Manifest Compliance (G3) | Not Started / Passed / Failed |
| U2.4 | U2 RFL Integrity Compliance (G4) | Not Started / Passed / Failed |
| U2.5 | U2 Uplift Curve Analysis (non-blocking) | Not Started / Positive / Mixed / Neutral / Negative |
```

These rows are:
- **U2.1–U2.4**: Compliance gates — MUST pass before U2.5 can be evaluated
- **U2.5**: Non-blocking evidence row — result informs but never blocks promotion

### Relationship to Original Uplift Conditions (C1–C7)

The Phase II U2 framework **extends but does not replace** the original 7 conditions:

| Original Condition | Phase II Extension |
|--------------------|--------------------|
| **C1** (Non-degenerate slice) | Satisfied by U2 asymmetric environment design |
| **C2** (Reproducible uplift) | Strengthened: requires 4 environments, not just 3 runs |
| **C3** (Clean manifests) | Formalized in Gate G3 |
| **C4** (ΔH attestation) | Formalized in Gate G2.4 |
| **C5** (Governance approval) | Unchanged — human sign-off still required |
| **C6** (Baseline on pre-promotion code) | Required by Gate G2.1 (fixed seeds) |
| **C7** (A/B measurement) | Satisfied by 4 asymmetric environments |

### Explicit Non-Blocking Guarantee

To prevent any ambiguity, the following guarantee is hereby established:

> **NON-BLOCKING GUARANTEE**
>
> Under NO circumstances shall U2 uplift evidence serve as a blocking criterion
> for basis promotion. The 22-step promotion criteria (D0.1–D3.7) and 18-rule
> purity contract remain the SOLE determinants of promotion eligibility.
>
> U2 evidence may:
> - Increase confidence in promotion decisions
> - Trigger investigations into unexpected patterns
> - Inform documentation and risk assessment
> - Provide supplementary audit trail
>
> U2 evidence may NOT:
> - Block promotion when all 22 steps pass
> - Override purity contract violations
> - Substitute for determinism verification
> - Replace human governance approval

### Governance Decision Points for U2 Evidence

The following decision points define when U2 evidence enters the governance workflow:

#### Decision Point DP1: Pre-Promotion Review

| Trigger | Action | Owner |
|---------|--------|-------|
| Promotion request received | Check if U2 runs exist for promoted modules | Governance Lead |
| U2 runs exist | Verify gates G1–G4 passed | Automated (validate-governance.ps1) |
| All gates passed | Include U2.5 result in promotion checklist | Governance Lead |
| Any gate failed | Note gate failure; proceed with promotion (non-blocking) | Governance Lead |

#### Decision Point DP2: Anomaly Investigation

| Trigger | Action | Owner |
|---------|--------|-------|
| U2.5 shows "Negative" | Create investigation ticket before proceeding | Governance Lead |
| Investigation complete | Document root cause in promotion notes | Investigator |
| Root cause is promoted code | STRONG recommendation to pause (still non-blocking) | Governance Lead |
| Root cause is external | Proceed with promotion; note in release | Governance Lead |

#### Decision Point DP3: Post-Promotion Monitoring

| Trigger | Action | Owner |
|---------|--------|-------|
| Promotion complete | Schedule U2 validation run on promoted codebase | CI Pipeline |
| U2 validation shows regression | Flag for next governance review | Automated |
| Regression confirmed | Consider rollback (separate process) | Governance Lead |

### PREREG_UPLIFT_U2.yaml Integration

The governance process MUST reference the preregistration file for:

1. **Experiment IDs**: All U2 runs must map to a defined `experiment_id`
2. **Slice Configs**: Each run must use the preregistered `slice_config` with matching hash
3. **Seeds**: Deterministic seeds must match preregistered values
4. **Success Metrics**: Only preregistered metrics may be used for uplift analysis

Any deviation from PREREG_UPLIFT_U2.yaml invalidates U2 evidence (Gate G1 failure).

### Compliance Gate Automation

The following scripts perform gate validation:

| Script | Gate | Execution |
|--------|------|-----------|
| `scripts/validate-governance.ps1` | G1–G4 | Manual or CI trigger |
| `scripts/validate-governance.sh` | G1–G4 | Unix equivalent |
| `tests/test_governance.py` | G1–G4 | pytest integration |

Script output must be logged to `ops/logs/u2_compliance.jsonl` per the logging schema above.

---

---

## Executive Summary

This document defines a **proposed promotion plan** for future Wave 1 work. It specifies:

1. A **rigorous decision tree** for evaluating module purity
2. The **complete module shortlist** with promotion classifications
3. A **formal basis purity contract** (18 enforceable rules)
4. A **4-day promotion plan** (Day 0–Day 3)
5. The **canonical import graph** with forbidden back-edges
6. A **test suite outline** for basis stability verification
7. A **risk register** with mitigations

**Note**: No modules have been promoted yet. When promotion begins, all modules
targeted for `basis/` must satisfy the purity contract without exception.

---

## 1. The Promotion Decision Tree

A module qualifies for `basis/` if and only if it passes ALL nodes in the following decision tree:

```
                    ┌──────────────────────────┐
                    │ START: Candidate Module  │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │ Q1: Is it PURE?          │
                    │ (no side effects,        │
                    │  no ambient reads)       │
                    └────────────┬─────────────┘
                           ┌─────┴─────┐
                       NO  │           │ YES
                           ▼           ▼
                    ┌──────────┐  ┌────────────────────┐
                    │ REJECT   │  │ Q2: Is it          │
                    │ (runtime │  │ DETERMINISTIC?     │
                    │  layer)  │  │ (same input →      │
                    └──────────┘  │  identical output) │
                                  └────────────┬───────┘
                                         ┌─────┴─────┐
                                     NO  │           │ YES
                                         ▼           ▼
                                  ┌──────────┐  ┌────────────────────┐
                                  │ REJECT   │  │ Q3: NO NETWORK?    │
                                  │ (non-    │  │ (no HTTP, no       │
                                  │  repro)  │  │  sockets, no DNS)  │
                                  └──────────┘  └────────────┬───────┘
                                                       ┌─────┴─────┐
                                                   NO  │           │ YES
                                                       ▼           ▼
                                                ┌──────────┐  ┌────────────────────┐
                                                │ REJECT   │  │ Q4: NO FILESYSTEM? │
                                                │ (I/O     │  │ (no open/read/     │
                                                │  bound)  │  │  write in logic)   │
                                                └──────────┘  └────────────┬───────┘
                                                                     ┌─────┴─────┐
                                                                 NO  │           │ YES
                                                                     ▼           ▼
                                                              ┌──────────┐  ┌────────────────────┐
                                                              │ REJECT   │  │ Q5: NO DATABASE?   │
                                                              │ (file    │  │ (no psycopg, no    │
                                                              │  I/O)    │  │  SQLAlchemy calls) │
                                                              └──────────┘  └────────────┬───────┘
                                                                                   ┌─────┴─────┐
                                                                               NO  │           │ YES
                                                                                   ▼           ▼
                                                                            ┌──────────┐  ┌────────────────────┐
                                                                            │ REJECT   │  │ Q6: NO REDIS?      │
                                                                            │ (DB      │  │ (no redis-py,      │
                                                                            │  bound)  │  │  no queue ops)     │
                                                                            └──────────┘  └────────────┬───────┘
                                                                                                 ┌─────┴─────┐
                                                                                             NO  │           │ YES
                                                                                                 ▼           ▼
                                                                                          ┌──────────┐  ┌────────────────────┐
                                                                                          │ REJECT   │  │ Q7: STDLIB ONLY?   │
                                                                                          │ (cache   │  │ (hashlib, json,    │
                                                                                          │  bound)  │  │  re, dataclasses)  │
                                                                                          └──────────┘  └────────────┬───────┘
                                                                                                               ┌─────┴─────┐
                                                                                                           NO* │           │ YES
                                                                                                               ▼           ▼
                                                                                                        ┌───────────┐  ┌─────────────┐
                                                                                                        │ REVIEW    │  │ Q8: NO      │
                                                                                                        │ EXCEPTION │  │ BACKEND/*   │
                                                                                                        │ (numpy    │  │ IMPORTS?    │
                                                                                                        │  allowed) │  └──────┬──────┘
                                                                                                        └───────────┘         │
                                                                                                                        ┌─────┴─────┐
                                                                                                                    NO  │           │ YES
                                                                                                                        ▼           ▼
                                                                                                                 ┌──────────┐ ┌──────────────┐
                                                                                                                 │ REJECT   │ │ ✓ PROMOTE    │
                                                                                                                 │ (import  │ │   TO BASIS/  │
                                                                                                                 │  cycle)  │ └──────────────┘
                                                                                                                 └──────────┘
```

### Purity Evaluation Checklist

For each candidate module, evaluate:

| Criterion | Test Method | Pass Condition |
|-----------|-------------|----------------|
| **Pure functions** | Static analysis: no `global`, no class state mutation | All exported functions are referentially transparent |
| **Deterministic hash boundaries** | Hash vector tests: `hash(x) == hash(x)` across runs | Identical inputs produce identical 64-char hex outputs |
| **Domain separation safety** | Audit domain tags: unique single-byte prefixes | LEAF=0x00, NODE=0x01, STMT=0x02, BLOCK=0x03 |
| **Canonical serialization stability** | RFC 8785 compliance: sorted keys, no whitespace | `rfc8785_canonicalize(x) == rfc8785_canonicalize(x)` |
| **Absence of ambient state** | Grep for `datetime.now`, `uuid.uuid4`, `random.*` | Zero hits in module source |
| **Forward-compatibility with RFL** | Type compatibility: accepts `DualAttestation`, `Block` | Interfaces align with `basis.core` types |
| **Forward-compatibility with Attestation Law** | Composite root formula: `H_t = SHA256(R_t || U_t)` | Verified against ATTESTATION_SPEC.md |

---

## 2. The Basis Wave-1 Module Shortlist

### 2.1 Law Domain (Attestation, Hash, Merkle)

| Module | Location | Classification | Rationale |
|--------|----------|----------------|-----------|
| `basis.core.types` | `basis/core/types.py` | **PROMOTE NOW** | Immutable frozen dataclasses (Block, DualAttestation, CurriculumTier), zero dependencies |
| `basis.crypto.hash` | `basis/crypto/hash.py` | **PROMOTE NOW** | Pure SHA-256 with domain separation, deterministic Merkle tree, no I/O |
| `basis.attestation.dual` | `basis/attestation/dual.py` | **PROMOTE NOW** | Pure composite root computation, hex validation, builds on `basis.crypto` |
| `attestation.dual_root` | `attestation/dual_root.py` | **PROMOTE AFTER REFACTOR** | RFC 8785 canonicalization, but imports `substrate.crypto.*` — consolidate |
| `substrate.crypto.hashing` | `substrate/crypto/hashing.py` | **DO NOT PROMOTE** | Duplicate of `basis.crypto.hash` — deprecate after migration |
| `substrate.crypto.core` | `substrate/crypto/core.py` | **PROMOTE AFTER REFACTOR** | RFC 8785 implementation needed, but Ed25519 signing is optional — extract pure subset |
| `tools/verify_merkle.py` | `tools/verify_merkle.py` | **DO NOT PROMOTE** | DB-bound verification tool — runtime layer only |

### 2.2 Economy Domain (Canon, Normalization, Derivation)

| Module | Location | Classification | Rationale |
|--------|----------|----------------|-----------|
| `basis.logic.normalizer` | `basis/logic/normalizer.py` | **PROMOTE NOW** | Pure normalization with LRU cache, ASCII-only output, deterministic |
| `normalization.canon` | `normalization/canon.py` | **PROMOTE AFTER REFACTOR** | Functional equivalent of `basis.logic.normalizer` — consolidate |
| `normalization.taut` | `normalization/taut.py` | **PROMOTE AFTER REFACTOR** | Pure truth table evaluation — move to `basis.logic.taut` |
| `normalization.truthtab` | `normalization/truthtab.py` | **PROMOTE AFTER REFACTOR** | Pure tautology checking — merge with `taut.py` |
| `backend.axiom_engine.axioms` | `backend/axiom_engine/axioms.py` | **PROMOTE AFTER REFACTOR** | Pure axiom schema definitions, but imports `normalization.canon` — redirect to basis |
| `backend.axiom_engine.bounds` | `backend/axiom_engine/bounds.py` | **PROMOTE NOW** | Pure dataclass (SliceBounds), stdlib only |
| `backend.axiom_engine.structure` | `backend/axiom_engine/structure.py` | **PROMOTE AFTER REFACTOR** | Pure formula structure analysis, but imports `normalization.canon` |
| `backend.axiom_engine.substitution` | `backend/axiom_engine/substitution.py` | **PROMOTE NOW** | Pure substitution logic, stdlib only |
| `backend.axiom_engine.derive_core` | `backend/axiom_engine/derive_core.py` | **DO NOT PROMOTE** | Has DB/Redis interactions — orchestrator layer |
| `basis.ledger.block` | `basis/ledger/block.py` | **PROMOTE NOW** | Pure block sealing with canonical JSON, builds on `basis.crypto` |

### 2.3 Metabolism Domain (RFL Pure Core)

| Module | Location | Classification | Rationale |
|--------|----------|----------------|-----------|
| `rfl.bootstrap_stats` | `rfl/bootstrap_stats.py` | **PROMOTE AFTER REFACTOR** | Pure numpy computation with deterministic seeds, but uses scipy — isolate core |
| `rfl.config` | `rfl/config.py` | **PROMOTE NOW** | Immutable dataclasses (RFLConfig, CurriculumSlice), stdlib only |
| `rfl.runner` | `rfl/runner.py` | **DO NOT PROMOTE** | Has Redis/file I/O — orchestrator layer |
| `rfl.coverage` | `rfl/coverage.py` | **DO NOT PROMOTE** | Loads baseline from DB — runtime layer |
| `rfl.audit` | `rfl/audit.py` | **PROMOTE AFTER REFACTOR** | Pure audit logging structures, but verify no I/O |
| `substrate.repro.determinism` | `substrate/repro/determinism.py` | **PROMOTE AFTER REFACTOR** | Pure determinism helpers, but optional numpy — ensure stdlib fallback |
| `basis.curriculum.ladder` | `basis/curriculum/ladder.py` | **PROMOTE NOW** | Pure curriculum tier management, file I/O isolated to optional loaders |

### Summary Table (PROPOSED — NOT EXECUTED)

| Classification | Count | Modules | Execution Status |
|----------------|-------|---------|------------------|
| **PROMOTE NOW** | 9 | `basis.core.types`, `basis.crypto.hash`, `basis.attestation.dual`, `basis.logic.normalizer`, `basis.ledger.block`, `basis.curriculum.ladder`, `backend.axiom_engine.bounds`, `backend.axiom_engine.substitution`, `rfl.config` | **Not Started** |
| **PROMOTE AFTER REFACTOR** | 9 | `attestation.dual_root`, `substrate.crypto.core`, `normalization.canon`, `normalization.taut`, `normalization.truthtab`, `backend.axiom_engine.axioms`, `backend.axiom_engine.structure`, `rfl.bootstrap_stats`, `substrate.repro.determinism` | **Not Started** |
| **DO NOT PROMOTE** | 6 | `substrate.crypto.hashing`, `tools/verify_merkle.py`, `backend.axiom_engine.derive_core`, `rfl.runner`, `rfl.coverage`, `rfl.audit` (pending review) | N/A |

**Note**: The classifications above are RECOMMENDATIONS. No promotion work has been executed.

---

## 3. Formal Basis Purity Contract

Every module in `basis/` MUST satisfy the following **18 rules**. Violations result in immediate promotion rejection or basis eviction.

### Category A: State Isolation (Rules 1–5)

| Rule | Statement | Enforcement |
|------|-----------|-------------|
| **A1** | **No Global State** — No module-level mutable variables, no `global` keyword, no class-level state that persists across calls | Static analysis: `grep -E "^[A-Za-z_]+ = \[\]" basis/` must return empty |
| **A2** | **No Ambient Reads** — Functions must not read environment variables, config files, or registry during execution | Static analysis: `grep -E "os\.(getenv|environ)" basis/` must return empty |
| **A3** | **No Wall-Clock Time** — No `datetime.now()`, `time.time()`, `time.perf_counter()` in business logic | Static analysis: `grep -E "datetime\.(now|utcnow)|time\.(time|perf_counter)" basis/` must return empty |
| **A4** | **No Random Entropy** — No `random.*`, `uuid.uuid4()`, `os.urandom` | Static analysis: `grep -E "random\.|uuid\.uuid4|os\.urandom" basis/` must return empty |
| **A5** | **No Nondeterministic Iteration** — All dict/set iterations must use `sorted()` or deterministic ordering | Code review: verify `for k in d:` patterns use `sorted(d.keys())` |

### Category B: Hash Contract Invariants (Rules 6–10)

| Rule | Statement | Enforcement |
|------|-----------|-------------|
| **B1** | **Lowercase Hex Only** — All hex digest outputs must be lowercase 64-character strings | Unit test: `assert all(c in '0123456789abcdef' for c in hash_output)` |
| **B2** | **Domain Separation Required** — Every SHA-256 call must include a domain prefix | Audit: no bare `hashlib.sha256(data)` without domain tag |
| **B3** | **Canonical Normalization Before Hashing** — All statement hashes must normalize input via `basis.logic.normalizer.normalize()` | Unit test: `hash_statement(x) == hash_statement(normalize(x))` |
| **B4** | **Deterministic Merkle Leaf Ordering** — Leaves must be sorted lexicographically by normalized content before tree construction | Unit test: `merkle_root([a, b]) == merkle_root([b, a])` |
| **B5** | **Hex Length Validation** — All root/digest parameters must validate `len(hex_str) == 64` before use | Code audit: composite_root validates both R_t and U_t lengths |

### Category C: Serialization Stability (Rules 11–14)

| Rule | Statement | Enforcement |
|------|-----------|-------------|
| **C1** | **RFC 8785 JSON Canonicalization** — All JSON serialization for hashing must use sorted keys, compact separators `(",", ":")`, no trailing whitespace | Unit test: `json.dumps(x, sort_keys=True, separators=(",", ":"))` |
| **C2** | **ASCII-Only Normalized Output** — Normalized formulas must contain only ASCII characters (0x00–0x7F) | Unit test: `normalized.encode("ascii")` must not raise |
| **C3** | **Immutable Data Structures** — All exported types must be frozen dataclasses or tuples | Type check: `@dataclass(frozen=True)` on all basis types |
| **C4** | **Stable Field Ordering** — Dataclass fields must be alphabetically ordered or follow a documented canonical order | Documentation requirement: field order documented in docstring |

### Category D: Import Graph Restrictions (Rules 15–18)

| Rule | Statement | Enforcement |
|------|-----------|-------------|
| **D1** | **No backend/* Imports** — Basis modules MUST NOT import from `backend.*` | Static analysis: `grep -r "from backend" basis/` must return empty |
| **D2** | **No substrate/* Imports (after consolidation)** — Post-Wave-1, basis modules must not import from `substrate.*` | Migration gate: all substrate functions absorbed into basis |
| **D3** | **No Circular Dependencies** — Import graph must be acyclic within basis | Import graph analysis: topological sort must succeed |
| **D4** | **No Runtime Dependencies** — Basis modules may only depend on stdlib, basis.*, and whitelisted pure libraries (numpy for stats only) | `pyproject.toml` audit: dependencies limited to stdlib + numpy |

---

## 4. Promotion Plan (Day 0 → Day 3) — PHASE II / NOT EXECUTED

> **⚠️ PHASE II WARNING**: This entire section describes PROPOSED future work.
> None of the tasks below have been started. The `substrate/` and `backend/`
> codepaths remain the authoritative runtime implementations.

### Day 0: Preparatory Hardening (NOT STARTED)

**Objective**: Consolidate duplicate implementations and confirm dual-root invariants.

| Task | Description | Verification |
|------|-------------|--------------|
| **D0.1** Hash Consolidation | Audit all SHA-256 call sites; redirect to `basis.crypto.hash` | `grep -r "hashlib.sha256" backend/ substrate/` shows only imports from basis |
| **D0.2** Canonical Normalize Unification | Confirm `basis.logic.normalizer.normalize()` and `normalization.canon.normalize()` produce identical output | Add test: `assert basis_normalize(x) == legacy_normalize(x)` for 1000 formulas |
| **D0.3** Dual-Root Invariant Confirmation | Verify `H_t = SHA256(R_t || U_t)` across all attestation code paths | Integration test: `test_first_organism.py` passes with deterministic H_t |
| **D0.4** Domain Tag Audit | Confirm unique domain tags across all modules | Document in HASHING_SPEC.md: LEAF=0x00, NODE=0x01, STMT=0x02, BLOCK=0x03 |
| **D0.5** Deprecation Notices | Add deprecation warnings to legacy modules | `warnings.warn("Use basis.X instead", DeprecationWarning)` |

### Day 1: Pure-Module Lift and Shim Installation (NOT STARTED)

**Objective**: Promote PROMOTE NOW modules; install import shims for backward compatibility.

| Task | Description | Verification |
|------|-------------|--------------|
| **D1.1** Verify Existing Basis | Run `test_basis_core.py` — confirm all primitives pass | `pytest tests/test_basis_core.py -v` all green |
| **D1.2** Install Import Shims | Create shims in deprecated locations redirecting to basis | `backend/logic/canon.py` → `from basis.logic.normalizer import normalize` |
| **D1.3** Update pyproject.toml | Add `basis` as explicit package; version bump to 0.2.0-wave1 | `pip install -e .` succeeds |
| **D1.4** Integration Smoke Test | Run First Organism integration test | `pytest tests/integration/test_first_organism.py -v` green |
| **D1.5** Determinism Verification | Run First Organism twice; compare H_t | `run1.H_t == run2.H_t` |

### Day 2: Integration of Attestation + Canon + Normalizer (NOT STARTED)

**Objective**: Promote PROMOTE AFTER REFACTOR modules after required changes.

| Task | Description | Verification |
|------|-------------|--------------|
| **D2.1** Consolidate Normalizers | Merge `normalization.canon` into `basis.logic.normalizer`; deprecate original | All imports redirect; unit tests pass |
| **D2.2** Consolidate Attestation | Merge `attestation.dual_root` pure functions into `basis.attestation.dual` | `generate_attestation_metadata()` available from basis |
| **D2.3** Extract RFC 8785 | Copy `rfc8785_canonicalize()` from `substrate.crypto.core` to `basis.crypto.json` | Pure module with no Ed25519 dependency |
| **D2.4** Extract Determinism Helpers | Copy pure functions from `substrate.repro.determinism` to `basis.repro.determinism` | Stdlib fallback when numpy unavailable |
| **D2.5** Full Test Suite | Run all unit and integration tests | `pytest --tb=short` all pass |

### Day 3: RFL Pure-Core Uplift and Final Import Graph Clean (NOT STARTED)

**Objective**: Complete promotion; verify clean import graph; tag release.

| Task | Description | Verification |
|------|-------------|--------------|
| **D3.1** RFL Bootstrap Stats Uplift | Extract pure numpy computation core to `basis.stats.bootstrap` | BCa CI computation available from basis |
| **D3.2** RFL Config Promotion | Move `rfl.config` dataclasses to `basis.rfl.config` | RFLConfig, CurriculumSlice importable from basis |
| **D3.3** Axiom Engine Pure Subset | Extract `bounds.py`, `substitution.py` to `basis.axiom.bounds`, `basis.axiom.substitution` | Pure modules with basis-only imports |
| **D3.4** Import Graph Validation | Run `test_basis_import_graph.py` | No forbidden back-edges detected |
| **D3.5** Purity Contract Audit | Run `test_basis_purity_contract.py` | All 18 rules pass |
| **D3.6** Tag Release | Create git tag `v0.2.0-wave1` | Tag points to passing CI |
| **D3.7** Documentation Update | Update README, CHANGELOG, this blueprint to COMPLETED | Markdown links verified |

---

## 5. Canonical Import Graph Diagram

### Target Import Graph (Post-Wave-1)

```
                              ┌─────────────────────────┐
                              │    EXTERNAL IMPORTS     │
                              │ (stdlib, numpy optional)│
                              └───────────┬─────────────┘
                                          │
       ┌──────────────────────────────────┼──────────────────────────────────┐
       │                                  │                                  │
       ▼                                  ▼                                  ▼
┌──────────────┐                ┌──────────────────┐               ┌──────────────────┐
│ basis.core   │                │ basis.repro      │               │ basis.stats      │
│   types.py   │                │   determinism.py │               │   bootstrap.py   │
│              │                │                  │               │                  │
│ Block        │                │ deterministic_   │               │ bootstrap_bca    │
│ BlockHeader  │                │   timestamp      │               │ compute_uplift_ci│
│ DualAttest.  │                │ deterministic_   │               │ compute_coverage │
│ CurriculumT. │                │   uuid           │               │   _ci            │
│ HexDigest    │                │ sorted_dict_     │               │ verify_          │
│ Normalized   │                │   items          │               │   metabolism     │
│   Formula    │                └──────────────────┘               └──────────────────┘
└──────┬───────┘
       │
       │ (types only, no logic)
       │
       ▼
┌──────────────────┐
│ basis.logic      │
│   normalizer.py  │
│                  │
│ normalize        │
│ are_equivalent   │
│ atoms            │
│ normalize_many   │
└────────┬─────────┘
         │
         │ (normalization)
         │
         ▼
┌──────────────────┐          ┌──────────────────┐
│ basis.crypto     │          │ basis.crypto     │
│   hash.py        │◄─────────│   json.py        │
│                  │          │                  │
│ sha256_hex       │          │ rfc8785_         │
│ sha256_bytes     │          │   canonicalize   │
│ hash_statement   │          └──────────────────┘
│ hash_block       │
│ merkle_root      │
│ compute_merkle_  │
│   proof          │
│ verify_merkle_   │
│   proof          │
│ reasoning_root   │
│ ui_root          │
│ DOMAIN_*         │
└────────┬─────────┘
         │
         │ (hashing + normalization)
         │
         ▼
┌──────────────────┐
│ basis.attestation│
│   dual.py        │
│                  │
│ reasoning_root   │
│ ui_root          │
│ composite_root   │
│ build_attest.    │
│ verify_attest.   │
│ attestation_     │
│   from_block     │
└────────┬─────────┘
         │
         │ (attestation + hash)
         │
         ▼
┌──────────────────┐
│ basis.ledger     │
│   block.py       │
│                  │
│ seal_block       │
│ block_to_dict    │
│ block_json       │
└──────────────────┘

┌──────────────────┐          ┌──────────────────┐
│ basis.axiom      │          │ basis.rfl        │
│   bounds.py      │          │   config.py      │
│   substitution.py│          │                  │
│                  │          │ RFLConfig        │
│ SliceBounds      │          │ CurriculumSlice  │
│ SubstitutionRule │          └──────────────────┘
└──────────────────┘

┌──────────────────┐
│ basis.curriculum │
│   ladder.py      │
│                  │
│ CurriculumLadder │
│ ladder_from_json │
│ ladder_to_json   │
└──────────────────┘
```

### Forbidden Back-Edges

The following imports are **STRICTLY FORBIDDEN** and will fail the import graph test:

| From | To | Reason |
|------|----|--------|
| `basis.*` | `backend.*` | Backend depends on basis, not reverse |
| `basis.*` | `substrate.*` | Substrate to be deprecated; basis must be self-contained |
| `basis.*` | `rfl.*` (non-basis) | RFL runner has I/O; only config can be promoted |
| `basis.*` | `attestation.*` (non-basis) | Attestation module to be consolidated into basis |
| `basis.*` | `ledger.*` (non-basis) | Ledger module has DB ops; only pure block logic in basis |
| `basis.*` | `normalization.*` | Normalization to be consolidated into basis.logic |

### Allowed External Consumers

```
backend.axiom_engine.*  ──┐
backend.ledger.*        ──┤
backend.crypto.*        ──┼──► basis.*  (ALLOWED: backend imports basis)
attestation.*           ──┤
rfl.*                   ──┤
substrate.*             ──┘
```

---

## 6. Basis Stability Test Suite Outline — PROPOSED / NOT IMPLEMENTED

> **Note**: The test files below are SKETCHES for future implementation.
> These tests DO NOT EXIST in the codebase. They are proposed as part of
> Phase II promotion work.

### test_basis_determinism.py (PROPOSED)

```python
"""Verify basis modules produce deterministic output across multiple runs."""

import pytest
from basis.logic.normalizer import normalize
from basis.crypto.hash import hash_statement, merkle_root
from basis.attestation.dual import composite_root, build_attestation
from basis.ledger.block import seal_block

@pytest.mark.parametrize("formula", [
    "p->p",
    "p -> q -> r",
    "(p -> q) -> r",
    "p /\\ q",
    "p \\/ q",
    "~p -> q",
])
def test_normalize_determinism(formula):
    """Same formula normalizes identically across runs."""
    results = [normalize(formula) for _ in range(100)]
    assert all(r == results[0] for r in results)

@pytest.mark.parametrize("statement", [
    "p->p",
    "p->q->r",
    "(p/\\q)\\/r",
])
def test_hash_statement_determinism(statement):
    """Same statement hashes identically across runs."""
    results = [hash_statement(statement) for _ in range(100)]
    assert all(r == results[0] for r in results)
    assert all(len(r) == 64 for r in results)

def test_merkle_root_order_independence():
    """Merkle root is identical regardless of input order."""
    leaves = ["p->p", "q->q", "r->r"]
    import itertools
    roots = [merkle_root(list(perm)) for perm in itertools.permutations(leaves)]
    assert all(r == roots[0] for r in roots)

def test_composite_root_determinism():
    """Composite root H_t is deterministic given R_t and U_t."""
    r_t = "a" * 64
    u_t = "b" * 64
    results = [composite_root(r_t, u_t) for _ in range(100)]
    assert all(r == results[0] for r in results)

def test_first_organism_determinism():
    """Full First Organism chain produces identical H_t across runs."""
    # Simplified test using basis primitives
    statements = ["p->p"]
    ui_events = ["select_statement"]

    results = []
    for _ in range(10):
        attestation = build_attestation(
            reasoning_events=statements,
            ui_events=ui_events
        )
        results.append(attestation.composite_root)

    assert all(r == results[0] for r in results)
```

### test_basis_hash_vectors.py

```python
"""Golden test vectors for hash functions."""

import pytest
from basis.crypto.hash import sha256_hex, hash_statement, merkle_root, DOMAIN_STMT, DOMAIN_LEAF

# Pre-computed test vectors (MUST NOT CHANGE after freeze)
HASH_VECTORS = {
    # Raw SHA-256 without domain (for reference only)
    "sha256_raw": [
        ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        ("p->p", "5d41402abc4b2a76b9719d911017c592"),  # example - compute actual
    ],
    # Statement hash with DOMAIN_STMT
    "hash_statement": [
        ("p->p", "<COMPUTED_VALUE>"),  # Fill in after freeze
        ("p->q->r", "<COMPUTED_VALUE>"),
        ("(p->q)->r", "<COMPUTED_VALUE>"),
    ],
    # Merkle roots
    "merkle_root": [
        (["p->p"], "<COMPUTED_VALUE>"),
        (["p->p", "q->q"], "<COMPUTED_VALUE>"),
        (["p->p", "q->q", "r->r"], "<COMPUTED_VALUE>"),
    ],
}

@pytest.mark.parametrize("statement,expected_hash", [
    ("p->p", "c3f5b3205153cf1c5a2b8a0c3694a7c3..."),  # First Organism canonical
])
def test_hash_statement_vector(statement, expected_hash):
    """Statement hash matches frozen test vector."""
    actual = hash_statement(statement)
    assert actual == expected_hash, f"Hash drift detected: {actual} != {expected_hash}"

def test_domain_tag_values():
    """Domain tags have correct byte values."""
    from basis.crypto.hash import DOMAIN_LEAF, DOMAIN_NODE, DOMAIN_STMT, DOMAIN_BLOCK
    assert DOMAIN_LEAF == b"\x00"
    assert DOMAIN_NODE == b"\x01"
    assert DOMAIN_STMT == b"\x02"
    assert DOMAIN_BLOCK == b"\x03"
```

### test_basis_attestation_vectors.py

```python
"""Golden test vectors for dual-root attestation (First Organism)."""

import pytest
from basis.attestation.dual import (
    reasoning_root,
    ui_root,
    composite_root,
    build_attestation,
)

# First Organism canonical values (from ATTESTATION_SPEC.md)
FIRST_ORGANISM_VECTORS = {
    "reasoning_root": "142a7c15ac8d44d2e13aa1006febdaa46f66cc4d69b89ebae5cba472e8b47902",
    "ui_root": "3c9a33d055fd8f7f61d2dbe1f9835889efea13427837a2213355f26495cfedee",
    "composite_root": "6a006e789be39105a37504e12305f5358baad273e15fb757f266c0f46b116e59",
}

def test_composite_root_formula():
    """H_t = SHA256(R_t || U_t) exactly."""
    import hashlib
    r_t = FIRST_ORGANISM_VECTORS["reasoning_root"]
    u_t = FIRST_ORGANISM_VECTORS["ui_root"]
    expected_h_t = FIRST_ORGANISM_VECTORS["composite_root"]

    # Manual computation
    manual = hashlib.sha256(f"{r_t}{u_t}".encode("ascii")).hexdigest()
    assert manual == expected_h_t

    # basis.attestation.dual computation
    computed = composite_root(r_t, u_t)
    assert computed == expected_h_t

def test_hex_validation():
    """composite_root rejects invalid hex inputs."""
    with pytest.raises(ValueError):
        composite_root("invalid", "a" * 64)
    with pytest.raises(ValueError):
        composite_root("a" * 63, "b" * 64)  # Wrong length
    with pytest.raises(ValueError):
        composite_root("g" * 64, "a" * 64)  # Invalid hex char
```

### test_basis_merkle_vectors.py

```python
"""Golden test vectors for Merkle tree construction."""

import pytest
from basis.crypto.hash import merkle_root, compute_merkle_proof, verify_merkle_proof

def test_empty_tree():
    """Empty tree has domain-separated empty sentinel."""
    root = merkle_root([])
    assert len(root) == 64
    # Value should be SHA256(DOMAIN_LEAF || "")

def test_single_leaf():
    """Single-leaf tree root equals leaf hash (domain-separated)."""
    leaves = ["p->p"]
    root = merkle_root(leaves)
    assert len(root) == 64

def test_two_leaves():
    """Two-leaf tree combines with NODE domain."""
    leaves = ["p->p", "q->q"]
    root = merkle_root(leaves)
    assert len(root) == 64

    # Verify order independence
    root_reversed = merkle_root(["q->q", "p->p"])
    assert root == root_reversed

def test_merkle_proof_roundtrip():
    """Proof generation and verification are consistent."""
    leaves = ["p->p", "q->q", "r->r"]
    root = merkle_root(leaves)

    for i, leaf in enumerate(leaves):
        proof = compute_merkle_proof(i, leaves)
        assert verify_merkle_proof(leaf, proof, root)

def test_tampered_proof_rejected():
    """Tampered proofs fail verification."""
    leaves = ["p->p", "q->q", "r->r"]
    root = merkle_root(leaves)
    proof = compute_merkle_proof(0, leaves)

    # Tamper with sibling hash
    tampered_proof = [(proof[0][0][:-1] + "x", proof[0][1])] + list(proof[1:])
    assert not verify_merkle_proof(leaves[0], tampered_proof, root)
```

### test_basis_import_graph.py

```python
"""Verify basis module import graph has no forbidden edges."""

import ast
import os
import pytest
from pathlib import Path

BASIS_ROOT = Path(__file__).parent.parent / "basis"
FORBIDDEN_IMPORTS = [
    "backend",
    "substrate",  # After consolidation
    "attestation",  # After consolidation (non-basis paths)
    "normalization",  # After consolidation
    "ledger",  # After consolidation (non-basis paths)
]

def get_imports(filepath: Path) -> list[str]:
    """Extract all import statements from a Python file."""
    with open(filepath, encoding="utf-8") as f:
        tree = ast.parse(f.read())

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports

def test_no_forbidden_imports():
    """No basis module imports from forbidden packages."""
    violations = []

    for pyfile in BASIS_ROOT.rglob("*.py"):
        if pyfile.name.startswith("__"):
            continue

        imports = get_imports(pyfile)
        for imp in imports:
            for forbidden in FORBIDDEN_IMPORTS:
                if imp.startswith(forbidden + ".") or imp == forbidden:
                    violations.append(f"{pyfile}: imports {imp}")

    if violations:
        pytest.fail("Forbidden imports detected:\n" + "\n".join(violations))

def test_no_circular_dependencies():
    """Basis import graph is acyclic."""
    # Build dependency graph
    deps = {}
    for pyfile in BASIS_ROOT.rglob("*.py"):
        if pyfile.name.startswith("__"):
            continue

        module_name = str(pyfile.relative_to(BASIS_ROOT.parent)).replace("/", ".").replace("\\", ".")[:-3]
        imports = get_imports(pyfile)
        basis_imports = [i for i in imports if i.startswith("basis.")]
        deps[module_name] = basis_imports

    # Kahn's algorithm for cycle detection
    in_degree = {m: 0 for m in deps}
    for m, dep_list in deps.items():
        for d in dep_list:
            if d in in_degree:
                in_degree[d] += 1

    queue = [m for m, d in in_degree.items() if d == 0]
    processed = 0

    while queue:
        m = queue.pop(0)
        processed += 1
        for dep in deps.get(m, []):
            if dep in in_degree:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

    if processed != len(deps):
        cycles = [m for m, d in in_degree.items() if d > 0]
        pytest.fail(f"Circular dependencies detected: {cycles}")
```

### test_basis_purity_contract.py

```python
"""Verify all 18 purity contract rules."""

import ast
import re
import pytest
from pathlib import Path

BASIS_ROOT = Path(__file__).parent.parent / "basis"

# Rule A1-A4: Forbidden patterns
FORBIDDEN_PATTERNS = [
    (r"datetime\.(now|utcnow)", "A3: Wall-clock time"),
    (r"time\.(time|perf_counter)", "A3: Wall-clock time"),
    (r"random\.", "A4: Random entropy"),
    (r"uuid\.uuid4", "A4: Random entropy"),
    (r"os\.urandom", "A4: Random entropy"),
    (r"os\.(getenv|environ)", "A2: Ambient reads"),
]

def test_no_forbidden_patterns():
    """Rules A1-A4: No forbidden runtime primitives."""
    violations = []

    for pyfile in BASIS_ROOT.rglob("*.py"):
        content = pyfile.read_text(encoding="utf-8")
        for pattern, rule in FORBIDDEN_PATTERNS:
            if re.search(pattern, content):
                violations.append(f"{pyfile}: {rule} (pattern: {pattern})")

    if violations:
        pytest.fail("Purity violations:\n" + "\n".join(violations))

def test_all_dataclasses_frozen():
    """Rule C3: All dataclasses are frozen."""
    violations = []

    for pyfile in BASIS_ROOT.rglob("*.py"):
        content = pyfile.read_text(encoding="utf-8")

        # Find @dataclass without frozen=True
        dataclass_pattern = r"@dataclass(?!\(.*frozen\s*=\s*True)"
        if re.search(dataclass_pattern, content):
            # Verify it's not just @dataclass(frozen=True) with different formatting
            if "@dataclass\n" in content or "@dataclass()" in content:
                violations.append(f"{pyfile}: Mutable dataclass detected")

    if violations:
        pytest.fail("Frozen dataclass violations:\n" + "\n".join(violations))

def test_hex_output_lowercase():
    """Rule B1: All hex outputs are lowercase."""
    # This would be verified via unit tests on actual outputs
    from basis.crypto.hash import sha256_hex

    test_cases = ["", "test", "p->p"]
    for tc in test_cases:
        result = sha256_hex(tc)
        assert result == result.lower(), f"Uppercase hex detected: {result}"
        assert len(result) == 64, f"Wrong hex length: {len(result)}"

def test_domain_separation_in_use():
    """Rule B2: Domain separation present in hash functions."""
    from basis.crypto import hash as crypto_hash

    # Verify domain constants exist
    assert hasattr(crypto_hash, "DOMAIN_LEAF")
    assert hasattr(crypto_hash, "DOMAIN_NODE")
    assert hasattr(crypto_hash, "DOMAIN_STMT")
    assert hasattr(crypto_hash, "DOMAIN_BLOCK")

    # Verify hash_statement uses DOMAIN_STMT
    # (Would require inspection or documentation verification)
```

---

## 7. Promotion Risk Register

### Risk 1: Duplicate Implementations

**Description**: Multiple implementations of normalizers, hashers, and attestation functions exist across `basis/`, `normalization/`, `substrate/`, and `backend/`.

**Impact**: High — Hash drift between implementations breaks ledger integrity.

**Probability**: Medium — Already partially consolidated but not complete.

**Mitigation**:
1. **Day 0**: Run cross-implementation equivalence tests
2. **Day 1**: Install import shims redirecting all legacy paths to basis
3. **Day 2**: Deprecate non-basis implementations with warnings
4. **Day 3**: Document single source of truth in HASHING_SPEC.md

**Owner**: Cursor P

### Risk 2: Implicit State in ui_event_store

**Description**: `backend/ledger/ui_events.py` uses thread-local storage for UI event capture.

**Impact**: Medium — Nondeterministic if accessed from multiple threads.

**Probability**: Low — First Organism path is single-threaded.

**Mitigation**:
1. Keep `ui_events.py` in runtime layer (DO NOT PROMOTE)
2. Require explicit `ui_events` parameter in `seal_block_with_dual_roots`
3. Document thread-safety requirements in ATTESTATION_SPEC.md

**Owner**: Integration test owner

### Risk 3: Noncanonical JSON Serialization in Old Modules

**Description**: Some modules use `json.dumps()` without `sort_keys=True` or with non-compact separators.

**Impact**: High — Attestation metadata hash drift.

**Probability**: Medium — Legacy code may have inconsistent JSON handling.

**Mitigation**:
1. **Day 0**: Grep for `json.dumps` without `sort_keys=True`
2. **Day 2**: Extract RFC 8785 canonicalizer to `basis.crypto.json`
3. Require all attestation JSON to flow through `rfc8785_canonicalize()`

**Owner**: Crypto module owner

### Risk 4: Merkle Verification Drift (tools/verify_merkle.py)

**Description**: `tools/verify_merkle.py` computes Merkle roots without domain separation, diverging from `basis.crypto.hash.merkle_root`.

**Impact**: High — Verification tool reports false failures.

**Probability**: High — Code inspection confirms no domain tags.

**Mitigation**:
1. **Day 0**: Add failing test comparing tool output to basis output
2. **Day 1**: Update tool to import from `basis.crypto.hash`
3. Add integration test running tool against recent blocks

**Owner**: Tools maintainer

### Risk 5: Determinism Helper Fragmentation

**Description**: `substrate/repro/determinism.py` has numpy dependency; some functions duplicate stdlib behavior.

**Impact**: Low — Numpy is optional but creates complexity.

**Probability**: Medium — Module has fallback paths but not tested thoroughly.

**Mitigation**:
1. **Day 2**: Extract pure stdlib subset to `basis.repro.determinism`
2. Keep numpy-dependent functions in `basis.stats.determinism` (optional)
3. Add test matrix: with numpy, without numpy

**Owner**: RFL module owner

### Risk 6: RFL Bootstrap Stats scipy Dependency

**Description**: `rfl/bootstrap_stats.py` requires scipy for `norm.ppf` inverse CDF.

**Impact**: Medium — Adds heavy dependency to basis.

**Probability**: Medium — scipy is common but not stdlib.

**Mitigation**:
1. Keep `bootstrap_stats.py` in `basis.stats` with scipy as optional extra
2. Document in pyproject.toml: `[basis.stats]` extra requires scipy
3. Provide fallback percentile bootstrap (no BCa) when scipy unavailable

**Owner**: RFL module owner

### Risk 7: First Organism Test Flakiness

**Description**: Integration test depends on exact H_t values; any code change causes failure.

**Impact**: Medium — Blocks CI on minor formatting changes.

**Probability**: Low — Canonical values frozen in ATTESTATION_SPEC.md.

**Mitigation**:
1. Document frozen test vectors in `artifacts/first_organism/`
2. Add "regenerate test vectors" script for intentional updates
3. Require explicit approval for any vector change

**Owner**: Test infrastructure owner

### Risk Summary Matrix

| Risk | Impact | Probability | Mitigation Status |
|------|--------|-------------|-------------------|
| R1: Duplicate Implementations | High | Medium | Day 0-2 consolidation |
| R2: Implicit ui_event_store State | Medium | Low | Keep in runtime layer |
| R3: Noncanonical JSON | High | Medium | RFC 8785 extraction |
| R4: Merkle Tool Drift | High | High | Tool update required |
| R5: Determinism Fragmentation | Low | Medium | Stdlib extraction |
| R6: scipy Dependency | Medium | Medium | Optional extra |
| R7: Test Flakiness | Medium | Low | Frozen vectors |

---

## Appendix A: First Organism Canonical Values

For reference, the canonical First Organism attestation values (frozen):

```json
{
  "reasoning_merkle_root": "142a7c15ac8d44d2e13aa1006febdaa46f66cc4d69b89ebae5cba472e8b47902",
  "ui_merkle_root": "3c9a33d055fd8f7f61d2dbe1f9835889efea13427837a2213355f26495cfedee",
  "composite_attestation_root": "6a006e789be39105a37504e12305f5358baad273e15fb757f266c0f46b116e59",
  "attestation_version": "v2",
  "algorithm": "SHA256",
  "composite_formula": "SHA256(R_t || U_t)"
}
```

---

## Appendix B: Domain Tag Registry

| Tag | Byte | Purpose | Used By |
|-----|------|---------|---------|
| DOMAIN_LEAF | 0x00 | Merkle tree leaf nodes | `basis.crypto.hash.merkle_root` |
| DOMAIN_NODE | 0x01 | Merkle tree internal nodes | `basis.crypto.hash.merkle_root` |
| DOMAIN_STMT | 0x02 | Statement content identity | `basis.crypto.hash.hash_statement` |
| DOMAIN_BLOCK | 0x03 | Block header identity | `basis.crypto.hash.hash_block` |
| DOMAIN_FED | 0x04 | Federation namespace | Reserved |
| DOMAIN_NODE_ATTEST | 0x05 | Node attestation namespace | Reserved |
| DOMAIN_DOSSIER | 0x06 | Celestial dossier namespace | Reserved |
| DOMAIN_ROOT | 0x07 | Root hash namespace | Reserved |
| DOMAIN_REASONING_EMPTY | 0x10 | Empty reasoning sentinel | `basis.crypto.hash.reasoning_root` |
| DOMAIN_UI_EMPTY | 0x11 | Empty UI sentinel | `basis.crypto.hash.ui_root` |

---

## Appendix C: Checklist for Cursor P

- [ ] Day 0 tasks complete (hash consolidation, normalize unification, dual-root confirmation)
- [ ] Day 1 tasks complete (shims installed, pyproject.toml updated, smoke tests pass)
- [ ] Day 2 tasks complete (normalizers consolidated, attestation merged, RFC 8785 extracted)
- [ ] Day 3 tasks complete (RFL core uplifted, import graph clean, purity contract verified)
- [ ] All test suites green (`test_basis_*.py`)
- [ ] Documentation updated (README, CHANGELOG, this blueprint marked COMPLETED)
- [ ] Git tag `v0.2.0-wave1` created
- [ ] CI passing on tag

---

**END OF WAVE1_PROMOTION_BLUEPRINT.md**

---

## ⚠️ REMINDER: THIS IS A BLUEPRINT ONLY ⚠️

*This document describes PROPOSED future work (Phase II).*

*No promotion steps have been executed. The `substrate/` and `backend/` codepaths
remain the authoritative runtime implementations. The `basis/` directory exists
but has NOT been formally promoted or validated per this plan.*

*This blueprint is NOT part of Evidence Pack v1.*
