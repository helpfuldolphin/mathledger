# Hₜ Series Governance Charter

> **STATUS: PHASE II — NOT RUN IN PHASE I**
>
> This charter defines the governance framework for Hₜ series creation, attestation
> sequencing, and cryptographic binding within MathLedger Phase II experiments.
>
> **Canonical Reference**: This document is normative for all U2 uplift experiments.
> Any deviation requires explicit governance review and charter amendment.
>
> **Related Documents**:
> - `HT_INVARIANT_SPEC_v1.md` — Technical invariant definitions
> - `PREREG_UPLIFT_U2.yaml` — Preregistration templates
> - `experiments/prereg/` — Experiment-specific preregistrations

---

## Document Control

| Field | Value |
|-------|-------|
| Charter Version | 1.1.0 |
| Effective Date | 2025-12-05 |
| Phase | II |
| Status | DRAFT — AWAITING GOVERNANCE APPROVAL |
| Last Reviewed | 2025-12-06 |
| Next Review | Before first U2 experiment execution |

### Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-05 | CLAUDE L | Initial charter creation |
| 1.1.0 | 2025-12-06 | CLAUDE L | Added §10 Replay-Hₜ Binding: INV-REPLAY-HT-* invariants, MDAP-Hₜ-Replay Triangle, failure semantics |

---

## Table of Contents

1. [Purpose and Scope](#1-purpose-and-scope)
2. [Definitions](#2-definitions)
3. [Governance Obligations](#3-governance-obligations)
4. [Attestation Sequencing Protocol](#4-attestation-sequencing-protocol)
5. [MDAP Binding Invariants](#5-mdap-binding-invariants)
6. [Non-Manipulability Constraints](#6-non-manipulability-constraints)
7. [Enforcement Mechanisms](#7-enforcement-mechanisms)
8. [Exception Handling](#8-exception-handling)
9. [Audit and Compliance](#9-audit-and-compliance)
10. [Replay-Hₜ Binding](#10-replay-hₜ-binding)
11. [Phase III Cryptographic Extensions](#11-phase-iii-cryptographic-extensions)
12. [Appendices](#12-appendices)

---

## 1. Purpose and Scope

### 1.1 Purpose

This charter establishes the governance framework ensuring that Hₜ series artifacts:

1. **Are deterministically reproducible** — Any party can independently verify the series
2. **Are cryptographically bound** — To preregistration, MDAP schedule, and experiment parameters
3. **Are tamper-evident** — Any modification is detectable
4. **Are non-repudiable** — The experiment executor cannot deny producing the series

### 1.2 Scope

This charter applies to:

- All Phase II U2 uplift experiments
- All `ht_series.json` artifacts produced by these experiments
- All verification processes applied to these artifacts
- All governance decisions regarding Hₜ series validity

This charter does NOT apply to:

- Phase I artifacts (governed by Phase I attestation rules)
- Non-U2 experiments (require separate governance charter)
- Observational metrics (ΔHₜ, correlation analyses) — these are non-normative

### 1.3 Authority

This charter derives authority from:

- The MathLedger Phase II specification
- The `HT_INVARIANT_SPEC_v1.md` technical standard
- The preregistration protocol (`PREREG_UPLIFT_U2.yaml`)

---

## 2. Definitions

### 2.1 Core Terms

| Term | Definition |
|------|------------|
| **Hₜ** | Per-cycle attestation hash binding proof root to cycle seed |
| **Hₜ series** | Ordered sequence of Hₜ values for all cycles in an experiment |
| **Hₜ chain** | Cumulative hash chain computed over the Hₜ series |
| **MDAP** | MathLedger Deterministic Attestation Protocol |
| **MDAP seed** | Root seed (0x4D444150) for all deterministic derivations |
| **Cycle seed** | Per-cycle seed derived from MDAP seed and cycle index |
| **Rₜ** | Proof Merkle root for a given cycle |
| **Binding hash** | Hash that cryptographically links two or more artifacts |

### 2.2 Roles

| Role | Responsibilities |
|------|------------------|
| **Experiment Executor** | Creates preregistration, runs experiment, produces artifacts |
| **Verifier** | Independently verifies all invariants hold |
| **Governance Authority** | Approves charter amendments, adjudicates disputes |
| **Auditor** | Conducts periodic compliance audits |

### 2.3 Document States

| State | Meaning |
|-------|---------|
| **DRAFT** | Under development, not yet binding |
| **APPROVED** | Binding for all covered experiments |
| **SUPERSEDED** | Replaced by newer version |
| **ARCHIVED** | Historical reference only |

---

## 3. Governance Obligations

### 3.1 Pre-Execution Obligations

Before any U2 experiment may begin execution, the Experiment Executor MUST:

#### 3.1.1 Preregistration Completion

```
GOV-PRE-1: Complete preregistration MUST exist before execution.
```

Requirements:
- Preregistration file MUST be committed to version control
- Preregistration MUST specify all success metric parameters
- Preregistration MUST NOT contain unresolved placeholders for:
  - `target_hashes` (if goal-based slice)
  - `min_goal_hits`, `min_total_verified`
  - `cycles` count

#### 3.1.2 MDAP Attestation

```
GOV-PRE-2: MDAP attestation MUST be computed and recorded before first cycle.
```

The MDAP attestation includes:
- MDAP seed (constant: 0x4D444150)
- Experiment ID
- Total cycle count
- Schedule hash (hash of all pre-computed cycle seeds)
- Attestation timestamp (UTC)

#### 3.1.3 Manifest Initialization

```
GOV-PRE-3: Manifest MUST be initialized with immutable fields before execution.
```

Immutable at initialization:
- `meta.version`
- `meta.phase`
- `experiment.id`
- `experiment.family`
- `preregistration.source_hash`
- `preregistration.binding_hash`
- `mdap_attestation.*`
- `success_metric.kind`
- `success_metric.parameters`
- `success_metric.parameters_hash`

### 3.2 During-Execution Obligations

#### 3.2.1 Determinism Enforcement

```
GOV-EXEC-1: All randomness MUST derive solely from cycle seeds.
```

Prohibited:
- System time as entropy source
- Hardware random number generators
- Network-derived randomness
- User input during execution

Permitted:
- `cycle_seed(c, experiment_id)` as PRNG seed
- Deterministic tie-breaking using cycle seed

#### 3.2.2 Per-Cycle Recording

```
GOV-EXEC-2: Each cycle MUST record all required fields before proceeding.
```

Required per-cycle:
- `cycle` (index)
- `R_t` (proof Merkle root)
- `H_t` (attestation hash)
- `cycle_seed` (for verification)
- `chain` (cumulative)
- `success` (boolean)
- `verified_count` (integer)

#### 3.2.3 No Retroactive Modification

```
GOV-EXEC-3: Once a cycle is recorded, its data MUST NOT be modified.
```

The Hₜ chain design ensures this cryptographically:
- Modifying Hₜ(c) invalidates chain(c) through chain(n-1)
- Checkpoints at ≤100 cycle intervals enable efficient detection

### 3.3 Post-Execution Obligations

#### 3.3.1 Series Finalization

```
GOV-POST-1: The ht_series.json file MUST be finalized within 60 seconds of last cycle.
```

Finalization includes:
- Computing `summary.chain_final`
- Computing `summary.ht_mdap_binding`
- Writing checkpoints
- Computing file hash

#### 3.3.2 Manifest Completion

```
GOV-POST-2: Manifest MUST be completed with all post-execution fields.
```

Required post-execution:
- `experiment.status` (complete or failed)
- `mdap_verification.post_execution_recomputed_hash`
- `mdap_verification.match`
- `artifacts.ht_series.hash`
- `ht_series.ht_chain_final`
- `ht_series.ht_mdap_binding`

#### 3.3.3 Verification Execution

```
GOV-POST-3: Full verification MUST pass before experiment is marked complete.
```

The verification suite (`verify_u2_experiment`) MUST:
- Pass all INV-PREREG-* checks
- Pass all INV-SUCCESS-* checks
- Pass all INV-MDAP-ATTEST-* checks
- Pass all INV-HT-CRYPTO-* checks
- Pass all INV-HTSERIES-* checks

#### 3.3.4 Artifact Preservation

```
GOV-POST-4: All artifacts MUST be preserved for minimum 2 years.
```

Required artifacts:
- `manifest.json`
- `ht_series.json`
- Experiment log (JSONL)
- Verification report
- Preregistration snapshot (at execution time)

---

## 4. Attestation Sequencing Protocol

### 4.1 Sequence Overview

The attestation sequence is strictly ordered. Violations invalidate the experiment.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ATTESTATION SEQUENCE TIMELINE                         │
└─────────────────────────────────────────────────────────────────────────────┘

T-0: PREREGISTRATION
│
├── Preregistration committed to VCS
├── Source hash computed: SHA256(PREREG_UPLIFT_U2.yaml)
└── Slice template selected

T-1: MANIFEST INITIALIZATION (must be before T-2)
│
├── Manifest created with meta, experiment, preregistration sections
├── Binding hash computed: SHA256(DOMAIN || source_hash || exp_id || timestamp)
└── Success metric parameters locked

T-2: MDAP ATTESTATION (must be before T-3)
│
├── Schedule hash computed over all n cycle seeds
├── MDAP attestation hash computed
├── Attestation recorded in manifest.mdap_attestation
└── Attestation timestamp recorded (marks T-2)

T-3: EXECUTION START (first cycle)
│
├── Cycle 0 begins
├── R_t(0) computed from proofs
├── H_t(0) = SHA256(DOMAIN || BE32(0) || R_t(0) || seed(0))
├── chain(0) = H_t(0)
└── Cycle 0 recorded to series

T-3+c: CYCLE c (for c = 1 to n-1)
│
├── R_t(c) computed
├── H_t(c) computed
├── chain(c) = SHA256(DOMAIN || chain(c-1) || H_t(c))
├── Checkpoint recorded if c mod 100 == 0
└── Cycle c recorded to series

T-4: EXECUTION END (after cycle n-1)
│
├── Last cycle completed
├── summary.chain_final = chain(n-1)
├── summary.ht_mdap_binding computed
└── ht_series.json written

T-5: VERIFICATION (must be before T-6)
│
├── All invariants verified
├── Verification report generated
└── Any failures recorded

T-6: FINALIZATION
│
├── manifest.experiment.status set
├── manifest.artifacts.ht_series.hash set
├── All artifacts preserved
└── Experiment marked EXECUTED in preregistration tracker
```

### 4.2 Timing Constraints

| Constraint | Requirement | Enforcement |
|------------|-------------|-------------|
| T-1 before T-2 | Manifest init before MDAP attestation | Timestamp comparison |
| T-2 before T-3 | MDAP attestation before first cycle | Timestamp in attestation vs first cycle |
| T-3 to T-4 continuous | No gaps > 1 hour between cycles | Cycle timestamps in log |
| T-4 to T-5 < 1 hour | Verification within 1 hour of completion | Report timestamp |
| T-5 to T-6 < 24 hours | Finalization within 24 hours | Manifest finalization timestamp |

### 4.3 Sequence Violation Handling

```
SEQ-FAIL-1: If T-2 timestamp >= T-3 timestamp (first cycle):
    → Experiment INVALID
    → Reason: MDAP_ATTESTATION_AFTER_EXECUTION_START

SEQ-FAIL-2: If manifest binding hash mismatch:
    → Experiment INVALID
    → Reason: PREREG_BINDING_MISMATCH

SEQ-FAIL-3: If cycle gap > 1 hour:
    → Experiment FLAGGED
    → Manual review required
    → Possible reasons: system failure, network issue

SEQ-FAIL-4: If verification not completed within 1 hour:
    → Experiment FLAGGED
    → Reason: DELAYED_VERIFICATION
    → Still valid if verification passes
```

---

## 5. MDAP Binding Invariants

### 5.1 MDAP Seed Invariant

```
MDAP-INV-1: The MDAP seed MUST be exactly 0x4D444150.
```

This constant is:
- ASCII encoding of "MDAP"
- 32-bit integer: 1296318800
- Chosen for:
  - Memorability
  - Non-zero (avoids edge cases)
  - Not a "magic number" that appears in common data

Verification:
```python
assert manifest["mdap_attestation"]["mdap_seed"] == "0x4D444150"
```

### 5.2 Cycle Seed Derivation Invariant

```
MDAP-INV-2: Cycle seed MUST be computed exactly as specified.
```

Formula:
```
cycle_seed(c, e) = SHA256(
    b"MathLedger:CycleSeed:v2:" ||
    BE32(0x4D444150) ||
    BE32(c) ||
    UTF8(e)
)
```

Where:
- `c` = cycle index (0-indexed)
- `e` = experiment ID (base form, without _baseline/_rfl suffix)
- `BE32` = 32-bit big-endian encoding
- `UTF8` = UTF-8 string encoding

### 5.3 Schedule Hash Invariant

```
MDAP-INV-3: Schedule hash MUST commit to all cycle seeds in order.
```

Formula:
```
schedule_hash(n, e) = SHA256(
    b"MathLedger:SeedSchedule:v2:" ||
    BE32(n) ||
    cycle_seed(0, e) ||
    cycle_seed(1, e) ||
    ... ||
    cycle_seed(n-1, e)
)
```

This hash:
- Commits to the total cycle count
- Commits to every seed in sequence
- Cannot be computed without knowing all seeds

### 5.4 MDAP Attestation Hash Invariant

```
MDAP-INV-4: MDAP attestation hash MUST include all protocol parameters.
```

Formula:
```
mdap_attestation_hash = SHA256(
    b"MathLedger:MDAPAttestation:v2:" ||
    BE32(MDAP_SEED) ||
    UTF8(experiment_id) ||
    BE32(total_cycles) ||
    schedule_hash ||
    UTF8(timestamp_utc)
)
```

### 5.5 Hₜ-MDAP Binding Invariant

```
MDAP-INV-5: The Hₜ chain MUST be cryptographically bound to MDAP attestation.
```

Formula:
```
ht_mdap_binding = SHA256(
    b"MathLedger:HtMdapBinding:v2:" ||
    chain_final ||
    mdap_attestation_hash
)
```

This binding proves:
- The Hₜ series was produced under the attested schedule
- The schedule was committed before execution began
- No schedule modification occurred during execution

### 5.6 Paired Run Invariant

```
MDAP-INV-6: Baseline and RFL runs MUST use identical cycle seeds.
```

For paired experiments:
```
cycle_seed(c, "uplift_u2_goal_001_baseline") uses base_id = "uplift_u2_goal_001"
cycle_seed(c, "uplift_u2_goal_001_rfl")      uses base_id = "uplift_u2_goal_001"
```

The `_baseline` and `_rfl` suffixes are stripped for seed computation.

---

## 6. Non-Manipulability Constraints

### 6.1 Overview

Non-manipulability ensures that:
1. The experiment executor cannot influence outcomes through parameter choices
2. The executor cannot selectively report favorable results
3. The executor cannot modify historical data

### 6.2 Pre-Commitment Constraints

#### 6.2.1 Target Hash Pre-Commitment

```
MANIP-PRE-1: Goal target hashes MUST be committed before execution.
```

For `slice_uplift_goal` experiments:
- `target_hashes` MUST be in preregistration
- MUST be committed to VCS before execution
- CANNOT be modified after `preregistration.status` = "EXECUTING"

Violation consequences:
- If targets are chosen after seeing results → INVALID
- If targets are modified during execution → INVALID

#### 6.2.2 Success Criteria Pre-Commitment

```
MANIP-PRE-2: Success metric parameters MUST be committed before execution.
```

Applies to:
- `min_goal_hits`
- `min_total_verified`
- `min_chain_length`
- `min_verified` (for density)
- `max_candidates` (for density)

#### 6.2.3 Cycle Count Pre-Commitment

```
MANIP-PRE-3: Total cycle count MUST be committed before execution.
```

The cycle count:
- Is recorded in MDAP attestation
- Cannot be changed without invalidating attestation hash
- Cannot be extended after seeing results

### 6.3 Execution-Time Constraints

#### 6.3.1 No Selective Recording

```
MANIP-EXEC-1: ALL cycles MUST be recorded, including failures.
```

Prohibited:
- Omitting cycles where success = false
- Recording only "interesting" cycles
- Filtering the series post-hoc

The Hₜ chain enforces this:
- Missing cycle c means chain(c) cannot be computed
- chain_final will not match if any cycle is omitted

#### 6.3.2 No Outcome-Dependent Termination

```
MANIP-EXEC-2: Experiment MUST run for exactly n cycles as pre-committed.
```

Prohibited:
- Stopping early after achieving "good enough" results
- Extending if results are unfavorable
- Restarting from a "better" seed

#### 6.3.3 No Parallel Experiments

```
MANIP-EXEC-3: Only ONE experiment per experiment_id may execute.
```

Prohibited:
- Running multiple instances with same experiment_id
- Cherry-picking the best result
- "Racing" parallel executions

Enforcement:
- Experiment_id includes timestamp
- MDAP attestation timestamp provides ordering
- Governance tracks active experiments

### 6.4 Post-Execution Constraints

#### 6.4.1 No Retroactive Analysis Changes

```
MANIP-POST-1: Statistical analysis method MUST match preregistration.
```

Locked at preregistration:
- `analysis.primary_metric`
- `analysis.secondary_metrics`
- `analysis.confidence_level`
- `analysis.ci_method`
- `analysis.test_method`

#### 6.4.2 No Selective Reporting

```
MANIP-POST-2: Full results MUST be reported regardless of outcome.
```

Required in report:
- Complete ht_series.json (all cycles)
- All verification results
- Success rate with confidence interval
- Comparison to null hypothesis

#### 6.4.3 No Data Modification

```
MANIP-POST-3: Artifacts MUST remain unmodified after finalization.
```

Enforced by:
- SHA-256 hash of ht_series.json in manifest
- SHA-256 hash of manifest in verification report
- Hₜ chain structure (any modification detectable)

### 6.5 Manipulation Detection

| Manipulation Attempt | Detection Method | Invariant |
|---------------------|------------------|-----------|
| Post-hoc target selection | Prereg hash mismatch | INV-PREREG-2 |
| Cycle seed manipulation | MDAP schedule mismatch | INV-MDAP-ATTEST-1 |
| Selective cycle recording | Chain verification failure | INV-HT-CHAIN-1 |
| Early termination | Cycle count mismatch | INV-SCHEDULE-1 |
| Result modification | File hash mismatch | INV-HTSERIES-1 |
| Checkpoint manipulation | Checkpoint verification | INV-HT-MANIFEST-2 |

---

## 7. Enforcement Mechanisms

### 7.1 Automated Enforcement

#### 7.1.1 Pre-Execution Gate

```python
def pre_execution_gate(manifest: dict, prereg_path: Path) -> GateResult:
    """
    MUST pass before execution may begin.

    Checks:
    - Preregistration exists and is committed
    - Manifest initialized with required fields
    - MDAP attestation computed
    - All binding hashes valid
    """
    checks = []

    # Check prereg exists
    if not prereg_path.exists():
        return GateResult(passed=False, reason="PREREG_NOT_FOUND")

    # Check prereg binding
    if not verify_prereg_binding(manifest):
        return GateResult(passed=False, reason="PREREG_BINDING_INVALID")

    # Check MDAP attestation
    if not verify_mdap_attestation(manifest):
        return GateResult(passed=False, reason="MDAP_ATTESTATION_INVALID")

    # Check success metric binding
    if not verify_success_params_hash(manifest):
        return GateResult(passed=False, reason="SUCCESS_METRIC_BINDING_INVALID")

    return GateResult(passed=True)
```

#### 7.1.2 Per-Cycle Enforcement

```python
def per_cycle_enforcement(
    cycle: int,
    entry: CycleEntry,
    expected_seed: bytes,
    prev_chain: bytes
) -> EnforcementResult:
    """
    MUST pass for each cycle before proceeding.

    Checks:
    - Cycle seed matches expected
    - H_t correctly computed
    - Chain correctly extended
    """
    # Verify seed
    if entry.cycle_seed != expected_seed:
        return EnforcementResult(
            passed=False,
            reason="CYCLE_SEED_MISMATCH",
            cycle=cycle
        )

    # Verify H_t
    expected_ht = compute_ht_cycle(cycle, entry.R_t, entry.cycle_seed)
    if entry.H_t != expected_ht:
        return EnforcementResult(
            passed=False,
            reason="HT_COMPUTATION_MISMATCH",
            cycle=cycle
        )

    # Verify chain
    if cycle == 0:
        expected_chain = entry.H_t
    else:
        expected_chain = compute_chain_step(prev_chain, entry.H_t)

    if entry.chain != expected_chain:
        return EnforcementResult(
            passed=False,
            reason="CHAIN_MISMATCH",
            cycle=cycle
        )

    return EnforcementResult(passed=True, cycle=cycle)
```

#### 7.1.3 Post-Execution Gate

```python
def post_execution_gate(
    manifest: dict,
    ht_series: dict,
    prereg: dict
) -> GateResult:
    """
    MUST pass before experiment may be marked complete.
    """
    result = verify_u2_experiment(manifest, ht_series, prereg)

    if not result.all_passed:
        return GateResult(
            passed=False,
            reason="VERIFICATION_FAILED",
            failed_invariants=result.failures
        )

    return GateResult(passed=True)
```

### 7.2 Manual Enforcement

#### 7.2.1 Governance Review Triggers

Manual review is triggered when:

| Trigger | Review Type | Authority |
|---------|-------------|-----------|
| First experiment of new slice type | Full review | Governance Authority |
| Cycle gap > 1 hour | Incident review | Auditor |
| Verification timeout | Delay review | Auditor |
| Charter exception request | Exception review | Governance Authority |
| Disputed result | Adjudication | Governance Authority |

#### 7.2.2 Review Process

```
1. Trigger identified
2. Review request submitted with:
   - Experiment ID
   - Trigger reason
   - All artifacts
   - Executor statement
3. Review conducted (within 5 business days)
4. Decision issued:
   - APPROVED: Experiment valid
   - REJECTED: Experiment invalid, with reasons
   - CONDITIONAL: Valid with noted caveats
5. Decision recorded in governance log
```

### 7.3 Enforcement Hierarchy

```
Level 1: Automated Gates
         │
         │ If gate fails → Experiment cannot proceed
         │
         ▼
Level 2: Verification Suite
         │
         │ If verification fails → Experiment marked FAILED
         │
         ▼
Level 3: Audit Review
         │
         │ Periodic or triggered reviews
         │
         ▼
Level 4: Governance Adjudication
         │
         │ For disputes and exceptions
         │
         ▼
Level 5: Charter Amendment
         │
         │ For systemic issues requiring rule changes
```

---

## 8. Exception Handling

### 8.1 Exception Categories

#### 8.1.1 Technical Exceptions

| Exception | Handling |
|-----------|----------|
| System crash during execution | May restart from checkpoint if < 100 cycles lost |
| Network failure | May pause and resume within 1 hour |
| Disk full | Must abort; cannot selectively delete |
| Verification timeout | May retry verification; original timestamps preserved |

#### 8.1.2 Governance Exceptions

| Exception | Handling |
|-----------|----------|
| Preregistration typo | May amend before execution with audit trail |
| Target hash unavailable | Must use substitute with governance approval |
| Cycle count change | NOT PERMITTED after MDAP attestation |
| Success metric change | NOT PERMITTED after manifest initialization |

### 8.2 Exception Request Process

```yaml
exception_request:
  experiment_id: "<experiment_id>"
  requested_by: "<executor_id>"
  request_date: "<ISO8601>"

  exception_type: "<technical|governance>"

  description: |
    Detailed description of the situation requiring exception.

  proposed_action: |
    What action is requested.

  justification: |
    Why this exception should be granted.

  impact_assessment:
    invariants_affected: ["INV-XXX-N", ...]
    risk_level: "<low|medium|high>"
    mitigation: |
      How risks will be mitigated.

  attachments:
    - error_logs.txt
    - system_state_snapshot.json
```

### 8.3 Exception Approval Criteria

An exception MAY be approved if:

1. **Non-manipulability preserved**: The exception cannot enable favorable result selection
2. **Verifiability preserved**: The exception does not prevent independent verification
3. **Audit trail exists**: All exception-related actions are logged
4. **Impact bounded**: The exception affects minimal invariants
5. **Precedent considered**: Similar exceptions have been granted OR novel situation justified

An exception MUST NOT be approved if:

1. It would allow post-hoc parameter changes
2. It would allow selective cycle recording
3. It would break the Hₜ chain integrity
4. It would invalidate the MDAP binding

---

## 9. Audit and Compliance

### 9.1 Audit Types

#### 9.1.1 Continuous Automated Audit

```python
class ContinuousAuditor:
    """
    Runs continuously, checking all new artifacts.
    """

    def audit_new_artifact(self, artifact_path: Path) -> AuditResult:
        """Called when new artifact is registered."""

        # Verify file hash matches manifest claim
        actual_hash = sha256_file(artifact_path)
        claimed_hash = self.get_claimed_hash(artifact_path)

        if actual_hash != claimed_hash:
            return AuditResult(
                status="ALERT",
                finding="FILE_HASH_MISMATCH",
                artifact=artifact_path
            )

        # Verify structure matches schema
        if not self.validate_schema(artifact_path):
            return AuditResult(
                status="ALERT",
                finding="SCHEMA_VIOLATION",
                artifact=artifact_path
            )

        return AuditResult(status="OK")
```

#### 9.1.2 Periodic Full Audit

Conducted quarterly or after every 10 experiments, whichever is sooner.

Scope:
- All experiments in period
- Random deep verification (20% of experiments)
- Cross-reference checks
- Anomaly detection

#### 9.1.3 Triggered Audit

Triggered by:
- Exception request
- Anomaly detection
- External report
- Governance request

### 9.2 Compliance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Invariant pass rate | 100% | Experiments passing all invariants |
| Exception rate | < 5% | Experiments requiring exceptions |
| Audit finding rate | < 1% | Audits finding issues |
| Mean time to verification | < 1 hour | Time from completion to verification |
| Artifact preservation | 100% | Required artifacts available at audit |

### 9.3 Audit Report Format

```json
{
  "audit": {
    "id": "AUDIT-2025-Q4-001",
    "type": "periodic",
    "period": {
      "start": "2025-10-01T00:00:00Z",
      "end": "2025-12-31T23:59:59Z"
    },
    "auditor": "<auditor_id>",
    "completed": "<ISO8601>"
  },
  "scope": {
    "experiments_total": 47,
    "experiments_audited": 47,
    "deep_verification_count": 10
  },
  "findings": {
    "critical": 0,
    "major": 0,
    "minor": 2,
    "informational": 5
  },
  "details": [
    {
      "severity": "minor",
      "experiment_id": "uplift_u2_goal_003",
      "finding": "Verification delayed by 73 minutes (target: 60)",
      "recommendation": "Improve verification pipeline capacity"
    }
  ],
  "compliance_summary": {
    "overall_status": "COMPLIANT",
    "invariant_compliance": "100%",
    "governance_compliance": "100%",
    "exception_rate": "2.1%"
  }
}
```

---

## 10. Replay-Hₜ Binding

> **STATUS: PHASE II — NOT RUN IN PHASE I**
>
> This section defines the cryptographic binding between Hₜ series and Replay Receipts,
> ensuring that replayed experiments produce verifiably identical attestation chains.

### 10.1 Overview

Replay verification is the cornerstone of determinism claims. When an experiment is replayed:
- The same MDAP seed schedule must produce the same cycle seeds
- The same cycle seeds with the same candidates must produce the same Rₜ values
- The same Rₜ values must produce the same Hₜ series
- The same Hₜ series must produce the same chain final and binding hashes

The **Replay Receipt** is the cryptographic proof that a replay was performed and matched
(or diverged from) the primary run.

### 10.2 Replay Receipt Structure

```json
{
  "meta": {
    "version": "2.0.0",
    "type": "replay_receipt",
    "generated_utc": "<ISO8601>"
  },
  "primary_run": {
    "experiment_id": "<experiment_id>",
    "manifest_hash": "<sha256 of primary manifest>",
    "ht_series_hash": "<sha256 of primary ht_series.json>",
    "chain_final": "<sha256 hex>",
    "ht_mdap_binding": "<sha256 hex>"
  },
  "replay_run": {
    "replay_id": "<replay_id>",
    "replay_timestamp_utc": "<ISO8601>",
    "ht_series_hash": "<sha256 of replay ht_series.json>",
    "chain_final": "<sha256 hex>",
    "ht_mdap_binding": "<sha256 hex>"
  },
  "verification": {
    "series_match": true,
    "chain_match": true,
    "binding_match": true,
    "cycle_divergence": null,
    "divergence_details": null
  },
  "binding": {
    "receipt_hash": "<sha256 of this receipt, excluding this field>",
    "primary_replay_binding": "<sha256(primary_chain || replay_chain)>"
  }
}
```

### 10.3 INV-REPLAY-HT Invariant Class

#### 10.3.1 INV-REPLAY-HT-1: Hₜ Series Identity

```
INV-REPLAY-HT-1: Primary and replay Hₜ series MUST be byte-identical.
```

**Definition**: Given:
- Primary Hₜ series file: `ht_series_primary.json`
- Replay Hₜ series file: `ht_series_replay.json`

The invariant holds iff:
```
SHA256(ht_series_primary.json) == SHA256(ht_series_replay.json)
```

**Rationale**: Determinism requires that identical inputs (MDAP seed, experiment parameters,
candidate set) produce identical outputs (Hₜ series). Any difference indicates:
- Non-deterministic behavior in the system
- External state leakage (time, random, network)
- Implementation bug

**Verification Pseudocode**:
```
FUNCTION verify_ht_series_identity(primary_path, replay_path) → RESULT
    INPUT:
        primary_path: filesystem path to primary ht_series.json
        replay_path: filesystem path to replay ht_series.json

    STEPS:
        1. primary_hash = SHA256(READ_BYTES(primary_path))
        2. replay_hash = SHA256(READ_BYTES(replay_path))
        3. IF primary_hash == replay_hash:
               RETURN RESULT(status=PASS, invariant="INV-REPLAY-HT-1")
           ELSE:
               RETURN RESULT(
                   status=FAIL,
                   invariant="INV-REPLAY-HT-1",
                   primary_hash=HEX(primary_hash),
                   replay_hash=HEX(replay_hash)
               )

    INVARIANT: INV-REPLAY-HT-1
END FUNCTION
```

#### 10.3.2 INV-REPLAY-HT-2: Chain Final Equivalence

```
INV-REPLAY-HT-2: Primary and replay chain_final MUST be identical.
```

**Definition**: Given:
- Primary chain final: `primary.ht_series.summary.chain_final`
- Replay chain final: `replay.ht_series.summary.chain_final`

The invariant holds iff:
```
primary.chain_final == replay.chain_final
```

**Rationale**: The chain final is the cumulative hash over all Hₜ values. Identity of
chain finals implies identity of all intermediate chain states, which implies identity
of all Hₜ values, which implies identity of all (Rₜ, cycle_seed) pairs.

**Verification Pseudocode**:
```
FUNCTION verify_chain_final_equivalence(primary_series, replay_series) → RESULT
    INPUT:
        primary_series: parsed primary ht_series.json
        replay_series: parsed replay ht_series.json

    STEPS:
        1. primary_chain = primary_series.summary.chain_final
        2. replay_chain = replay_series.summary.chain_final
        3. IF primary_chain == replay_chain:
               RETURN RESULT(status=PASS, invariant="INV-REPLAY-HT-2")
           ELSE:
               # Find first divergence point
               divergence = find_first_divergence(
                   primary_series.series,
                   replay_series.series
               )
               RETURN RESULT(
                   status=FAIL,
                   invariant="INV-REPLAY-HT-2",
                   primary_chain=primary_chain,
                   replay_chain=replay_chain,
                   first_divergence_cycle=divergence.cycle
               )

    INVARIANT: INV-REPLAY-HT-2
END FUNCTION

FUNCTION find_first_divergence(primary_entries, replay_entries) → DIVERGENCE
    FOR i = 0 TO MIN(LEN(primary_entries), LEN(replay_entries)) - 1:
        IF primary_entries[i].H_t != replay_entries[i].H_t:
            RETURN DIVERGENCE(
                cycle=i,
                primary_ht=primary_entries[i].H_t,
                replay_ht=replay_entries[i].H_t,
                primary_rt=primary_entries[i].R_t,
                replay_rt=replay_entries[i].R_t
            )
    IF LEN(primary_entries) != LEN(replay_entries):
        RETURN DIVERGENCE(
            cycle=MIN(LEN(primary_entries), LEN(replay_entries)),
            reason="LENGTH_MISMATCH"
        )
    RETURN NULL  # No divergence
END FUNCTION
```

#### 10.3.3 INV-REPLAY-HT-3: MDAP Binding Preservation

```
INV-REPLAY-HT-3: Primary and replay ht_mdap_binding MUST be identical.
```

**Definition**: Given:
- Primary binding: `primary.ht_series.summary.ht_mdap_binding`
- Replay binding: `replay.ht_series.summary.ht_mdap_binding`

The invariant holds iff:
```
primary.ht_mdap_binding == replay.ht_mdap_binding
```

**Rationale**: The MDAP binding cryptographically links the Hₜ chain to the MDAP attestation.
If bindings match, it proves both runs used the same MDAP attestation (same seed schedule)
and produced the same Hₜ chain.

**Verification Pseudocode**:
```
FUNCTION verify_mdap_binding_preservation(primary_series, replay_series, manifest) → RESULT
    INPUT:
        primary_series: parsed primary ht_series.json
        replay_series: parsed replay ht_series.json
        manifest: parsed experiment manifest (shared by both runs)

    CONSTANTS:
        DOMAIN = b"MathLedger:HtMdapBinding:v2:"

    STEPS:
        1. primary_binding = primary_series.summary.ht_mdap_binding
        2. replay_binding = replay_series.summary.ht_mdap_binding

        # Both must match each other
        3. IF primary_binding != replay_binding:
               RETURN RESULT(
                   status=FAIL,
                   invariant="INV-REPLAY-HT-3",
                   reason="BINDING_MISMATCH",
                   primary_binding=primary_binding,
                   replay_binding=replay_binding
               )

        # Both must be correctly computed
        4. mdap_attestation_hash = manifest.mdap_attestation.attestation_hash
        5. expected_binding = SHA256(
               DOMAIN ||
               DECODE_HEX(primary_series.summary.chain_final) ||
               DECODE_HEX(mdap_attestation_hash)
           )
        6. IF HEX(expected_binding) != primary_binding:
               RETURN RESULT(
                   status=FAIL,
                   invariant="INV-REPLAY-HT-3",
                   reason="BINDING_COMPUTATION_ERROR",
                   expected=HEX(expected_binding),
                   actual=primary_binding
               )

        7. RETURN RESULT(status=PASS, invariant="INV-REPLAY-HT-3")

    INVARIANT: INV-REPLAY-HT-3
END FUNCTION
```

#### 10.3.4 INV-REPLAY-HT-4: Replay Receipt Integrity

```
INV-REPLAY-HT-4: Replay Receipt MUST correctly bind primary and replay artifacts.
```

**Definition**: The replay receipt must:
1. Contain correct hashes of both Hₜ series files
2. Contain correct chain finals from both series
3. Contain correct binding hashes from both series
4. Have a self-consistent receipt hash

**Verification Pseudocode**:
```
FUNCTION verify_replay_receipt_integrity(
    receipt,
    primary_series_path,
    replay_series_path
) → RESULT
    INPUT:
        receipt: parsed replay_receipt.json
        primary_series_path: path to primary ht_series.json
        replay_series_path: path to replay ht_series.json

    STEPS:
        # Verify primary series hash
        1. actual_primary_hash = SHA256(READ_BYTES(primary_series_path))
        2. IF HEX(actual_primary_hash) != receipt.primary_run.ht_series_hash:
               RETURN RESULT(
                   status=FAIL,
                   invariant="INV-REPLAY-HT-4",
                   reason="PRIMARY_SERIES_HASH_MISMATCH"
               )

        # Verify replay series hash
        3. actual_replay_hash = SHA256(READ_BYTES(replay_series_path))
        4. IF HEX(actual_replay_hash) != receipt.replay_run.ht_series_hash:
               RETURN RESULT(
                   status=FAIL,
                   invariant="INV-REPLAY-HT-4",
                   reason="REPLAY_SERIES_HASH_MISMATCH"
               )

        # Verify chain finals match series contents
        5. primary_series = PARSE_JSON(READ(primary_series_path))
        6. replay_series = PARSE_JSON(READ(replay_series_path))
        7. IF receipt.primary_run.chain_final != primary_series.summary.chain_final:
               RETURN RESULT(
                   status=FAIL,
                   invariant="INV-REPLAY-HT-4",
                   reason="PRIMARY_CHAIN_MISMATCH"
               )
        8. IF receipt.replay_run.chain_final != replay_series.summary.chain_final:
               RETURN RESULT(
                   status=FAIL,
                   invariant="INV-REPLAY-HT-4",
                   reason="REPLAY_CHAIN_MISMATCH"
               )

        # Verify receipt self-hash
        9. receipt_for_hash = COPY(receipt)
        10. DELETE receipt_for_hash.binding.receipt_hash
        11. expected_receipt_hash = SHA256(CANONICAL_JSON(receipt_for_hash))
        12. IF HEX(expected_receipt_hash) != receipt.binding.receipt_hash:
               RETURN RESULT(
                   status=FAIL,
                   invariant="INV-REPLAY-HT-4",
                   reason="RECEIPT_HASH_MISMATCH"
               )

        13. RETURN RESULT(status=PASS, invariant="INV-REPLAY-HT-4")

    INVARIANT: INV-REPLAY-HT-4
END FUNCTION
```

#### 10.3.5 INV-REPLAY-HT-5: Primary-Replay Binding Hash

```
INV-REPLAY-HT-5: The primary-replay binding MUST cryptographically link both chain finals.
```

**Definition**: The `primary_replay_binding` field must equal:
```
SHA256(b"MathLedger:PrimaryReplayBinding:v2:" || primary_chain_final || replay_chain_final)
```

**Verification Pseudocode**:
```
FUNCTION verify_primary_replay_binding(receipt) → RESULT
    INPUT:
        receipt: parsed replay_receipt.json

    CONSTANTS:
        DOMAIN = b"MathLedger:PrimaryReplayBinding:v2:"

    STEPS:
        1. primary_chain = DECODE_HEX(receipt.primary_run.chain_final)
        2. replay_chain = DECODE_HEX(receipt.replay_run.chain_final)
        3. expected_binding = SHA256(DOMAIN || primary_chain || replay_chain)
        4. IF HEX(expected_binding) != receipt.binding.primary_replay_binding:
               RETURN RESULT(
                   status=FAIL,
                   invariant="INV-REPLAY-HT-5",
                   expected=HEX(expected_binding),
                   actual=receipt.binding.primary_replay_binding
               )
        5. RETURN RESULT(status=PASS, invariant="INV-REPLAY-HT-5")

    INVARIANT: INV-REPLAY-HT-5
END FUNCTION
```

### 10.4 MDAP-Hₜ-Replay Triangle

The MDAP-Hₜ-Replay Triangle establishes the cryptographic relationships between
the three core artifacts of a verifiable experiment.

#### 10.4.1 Triangle Diagram

```
                            ┌─────────────────────────────────────┐
                            │         MDAP ATTESTATION            │
                            │                                     │
                            │  mdap_attestation_hash = SHA256(    │
                            │      DOMAIN_MDAP_ATTEST ||          │
                            │      BE32(MDAP_SEED) ||             │
                            │      experiment_id ||               │
                            │      BE32(total_cycles) ||          │
                            │      schedule_hash ||               │
                            │      timestamp_utc                  │
                            │  )                                  │
                            └──────────────┬──────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    │              schedule_hash                  │
                    │                      │                      │
                    │     Commits to all   │   cycle_seed(c)      │
                    │     n cycle seeds    │   derived from       │
                    │                      │   MDAP_SEED + c      │
                    │                      │                      │
                    ▼                      ▼                      ▼
┌───────────────────────────┐    ┌───────────────────────────┐
│     PRIMARY Hₜ SERIES     │    │      REPLAY Hₜ SERIES     │
│                           │    │                           │
│  For each cycle c:        │    │  For each cycle c:        │
│    seed(c) = f(MDAP, c)   │    │    seed(c) = f(MDAP, c)   │
│    H_t(c) = g(R_t, seed)  │    │    H_t(c) = g(R_t, seed)  │
│    chain(c) = h(prev, H_t)│    │    chain(c) = h(prev, H_t)│
│                           │    │                           │
│  chain_final_primary      │    │  chain_final_replay       │
└─────────────┬─────────────┘    └─────────────┬─────────────┘
              │                                │
              │     ┌──────────────────┐       │
              │     │  MUST BE EQUAL   │       │
              │     │  (INV-REPLAY-HT-2)│      │
              └────►│                  │◄──────┘
                    └────────┬─────────┘
                             │
                             │ If equal, then:
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│  ht_mdap_binding_primary│   │  ht_mdap_binding_replay │
│  = SHA256(              │   │  = SHA256(              │
│      DOMAIN_BIND ||     │   │      DOMAIN_BIND ||     │
│      chain_final ||     │   │      chain_final ||     │
│      mdap_attest_hash   │   │      mdap_attest_hash   │
│  )                      │   │  )                      │
└───────────┬─────────────┘   └─────────────┬───────────┘
            │                               │
            │      ┌──────────────────┐     │
            │      │  MUST BE EQUAL   │     │
            └─────►│ (INV-REPLAY-HT-3)│◄────┘
                   └────────┬─────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │       REPLAY RECEIPT        │
              │                             │
              │  primary_replay_binding =   │
              │    SHA256(                  │
              │      DOMAIN_PR_BIND ||      │
              │      chain_final_primary || │
              │      chain_final_replay     │
              │    )                        │
              │                             │
              │  receipt_hash = SHA256(     │
              │    canonical_receipt        │
              │  )                          │
              └─────────────────────────────┘
```

#### 10.4.2 Hash Equality Requirements

| Relationship | Hashes That MUST Equal | Invariant |
|--------------|------------------------|-----------|
| Primary ↔ Replay series | `SHA256(ht_series_primary.json)` = `SHA256(ht_series_replay.json)` | INV-REPLAY-HT-1 |
| Primary ↔ Replay chain | `chain_final_primary` = `chain_final_replay` | INV-REPLAY-HT-2 |
| Primary ↔ Replay binding | `ht_mdap_binding_primary` = `ht_mdap_binding_replay` | INV-REPLAY-HT-3 |
| Receipt ↔ Artifacts | Receipt hashes match artifact hashes | INV-REPLAY-HT-4 |

#### 10.4.3 Hash Traceability Requirements

| From | To | Traceability Path |
|------|-----|-------------------|
| MDAP attestation | Cycle seeds | `mdap_attestation_hash` contains `schedule_hash` which commits to all `cycle_seed(c)` |
| Cycle seeds | Hₜ values | Each `H_t(c) = SHA256(DOMAIN || c || R_t(c) || cycle_seed(c))` |
| Hₜ values | Chain final | `chain(n-1)` is cumulative hash over all `H_t` |
| Chain final | MDAP binding | `ht_mdap_binding = SHA256(DOMAIN || chain_final || mdap_attestation_hash)` |
| Both chains | Replay binding | `primary_replay_binding = SHA256(DOMAIN || chain_primary || chain_replay)` |

#### 10.4.4 Triangle Verification Algorithm

```
FUNCTION verify_mdap_ht_replay_triangle(
    manifest,
    primary_series,
    replay_series,
    receipt
) → RESULT
    INPUT:
        manifest: parsed experiment manifest
        primary_series: parsed primary ht_series.json
        replay_series: parsed replay ht_series.json
        receipt: parsed replay_receipt.json

    STEPS:
        # === MDAP VERTEX ===
        # Verify MDAP attestation is valid
        1. IF NOT verify_mdap_attestation(manifest):
               RETURN RESULT(status=FAIL, vertex="MDAP", reason="INVALID_ATTESTATION")

        # === PRIMARY Hₜ VERTEX ===
        # Verify primary series chain is valid
        2. IF NOT verify_ht_series_chain(primary_series):
               RETURN RESULT(status=FAIL, vertex="PRIMARY_HT", reason="INVALID_CHAIN")

        # Verify primary MDAP binding
        3. IF NOT verify_ht_mdap_binding(primary_series, manifest):
               RETURN RESULT(status=FAIL, vertex="PRIMARY_HT", reason="INVALID_MDAP_BINDING")

        # === REPLAY Hₜ VERTEX ===
        # Verify replay series chain is valid
        4. IF NOT verify_ht_series_chain(replay_series):
               RETURN RESULT(status=FAIL, vertex="REPLAY_HT", reason="INVALID_CHAIN")

        # Verify replay MDAP binding
        5. IF NOT verify_ht_mdap_binding(replay_series, manifest):
               RETURN RESULT(status=FAIL, vertex="REPLAY_HT", reason="INVALID_MDAP_BINDING")

        # === EDGE: PRIMARY ↔ REPLAY ===
        # Verify series identity (INV-REPLAY-HT-1)
        6. result_1 = verify_ht_series_identity(primary_series, replay_series)
        7. IF result_1.status == FAIL:
               RETURN result_1

        # Verify chain equivalence (INV-REPLAY-HT-2)
        8. result_2 = verify_chain_final_equivalence(primary_series, replay_series)
        9. IF result_2.status == FAIL:
               RETURN result_2

        # Verify MDAP binding preservation (INV-REPLAY-HT-3)
        10. result_3 = verify_mdap_binding_preservation(primary_series, replay_series, manifest)
        11. IF result_3.status == FAIL:
               RETURN result_3

        # === RECEIPT VERTEX ===
        # Verify receipt integrity (INV-REPLAY-HT-4)
        12. result_4 = verify_replay_receipt_integrity(receipt, primary_series, replay_series)
        13. IF result_4.status == FAIL:
               RETURN result_4

        # Verify primary-replay binding (INV-REPLAY-HT-5)
        14. result_5 = verify_primary_replay_binding(receipt)
        15. IF result_5.status == FAIL:
               RETURN result_5

        # === ALL CHECKS PASSED ===
        16. RETURN RESULT(
               status=PASS,
               triangle_valid=TRUE,
               invariants_checked=[
                   "INV-REPLAY-HT-1",
                   "INV-REPLAY-HT-2",
                   "INV-REPLAY-HT-3",
                   "INV-REPLAY-HT-4",
                   "INV-REPLAY-HT-5"
               ]
           )
END FUNCTION
```

### 10.5 Failure Semantics

#### 10.5.1 Failure Severity Classification

| Invariant | Failure Severity | Impact |
|-----------|------------------|--------|
| INV-REPLAY-HT-1 | **CRITICAL** | Run INVALID |
| INV-REPLAY-HT-2 | **CRITICAL** | Run INVALID |
| INV-REPLAY-HT-3 | **CRITICAL** | Run INVALID |
| INV-REPLAY-HT-4 | **HIGH** | Requires manual investigation |
| INV-REPLAY-HT-5 | **HIGH** | Requires manual investigation |

#### 10.5.2 Failure Definitions

##### INV-REPLAY-HT-1 Failure: Series Identity Mismatch

**Severity**: CRITICAL

**Impact**: Run INVALID — The experiment cannot claim determinism.

**Cause Analysis**:
- Non-deterministic behavior in proof generation
- External state dependency (time, random, network)
- Different candidate sets between runs
- Implementation bug in Rₜ computation

**Required Actions**:
1. Experiment is marked `status: FAILED`
2. Failure reason: `REPLAY_SERIES_MISMATCH`
3. Both series must be preserved for forensic analysis
4. Manual root cause analysis required before re-run
5. No claims about experiment validity may be made

**Remediation Path**:
```
1. Identify divergence cycle via find_first_divergence()
2. Compare R_t values at divergence cycle
3. If R_t differs: investigate proof generation
4. If R_t matches but H_t differs: investigate hash computation
5. Fix root cause
6. Re-run both primary and replay from scratch
```

##### INV-REPLAY-HT-2 Failure: Chain Final Mismatch

**Severity**: CRITICAL

**Impact**: Run INVALID — The cumulative attestation chains diverged.

**Cause Analysis**:
- If INV-REPLAY-HT-1 also failed: root cause is series divergence
- If INV-REPLAY-HT-1 passed: impossible state (bug in verification logic)

**Required Actions**:
1. Experiment is marked `status: FAILED`
2. Failure reason: `REPLAY_CHAIN_MISMATCH`
3. If INV-REPLAY-HT-1 passed, escalate as verification bug

##### INV-REPLAY-HT-3 Failure: MDAP Binding Mismatch

**Severity**: CRITICAL

**Impact**: Run INVALID — The Hₜ-MDAP binding is inconsistent.

**Cause Analysis**:
- Different MDAP attestation used between runs
- Chain final mismatch (implies INV-REPLAY-HT-2 also failed)
- Bug in binding computation

**Required Actions**:
1. Experiment is marked `status: FAILED`
2. Failure reason: `REPLAY_MDAP_BINDING_MISMATCH`
3. Verify both runs used same manifest
4. Verify MDAP attestation hash consistency

##### INV-REPLAY-HT-4 Failure: Receipt Integrity Error

**Severity**: HIGH

**Impact**: Requires manual investigation — Receipt may be corrupted or tampered.

**Cause Analysis**:
- Receipt generated from wrong artifacts
- Receipt modified after generation
- Hash computation bug
- File corruption

**Required Actions**:
1. Experiment is marked `status: UNDER_REVIEW`
2. Failure reason: `REPLAY_RECEIPT_INTEGRITY_ERROR`
3. Manual comparison of receipt fields vs actual artifacts
4. If artifacts match but receipt wrong: regenerate receipt
5. If artifacts don't match receipt: investigate artifact provenance

**Remediation Path**:
```
1. Verify artifact files exist and are readable
2. Recompute all hashes from artifacts
3. Compare with receipt claims
4. If discrepancy in receipt generation: fix generator, regenerate
5. If discrepancy in artifacts: treat as potential tampering, escalate
```

##### INV-REPLAY-HT-5 Failure: Primary-Replay Binding Error

**Severity**: HIGH

**Impact**: Requires manual investigation — Binding computation may be incorrect.

**Cause Analysis**:
- Bug in binding hash computation
- Wrong domain prefix used
- Receipt corruption

**Required Actions**:
1. Experiment is marked `status: UNDER_REVIEW`
2. Failure reason: `REPLAY_PRIMARY_BINDING_ERROR`
3. Manual recomputation of binding hash
4. If computation matches expected: receipt corruption
5. If computation differs: fix computation logic

#### 10.5.3 Failure Response Matrix

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FAILURE RESPONSE MATRIX                               │
├─────────────────┬──────────────┬─────────────────┬──────────────────────────┤
│ Invariant       │ Severity     │ Auto-Response   │ Human Action Required    │
├─────────────────┼──────────────┼─────────────────┼──────────────────────────┤
│ INV-REPLAY-HT-1 │ CRITICAL     │ Mark FAILED     │ Root cause analysis      │
│                 │              │ Block claims    │ Fix determinism issue    │
│                 │              │ Preserve logs   │ Re-run experiment        │
├─────────────────┼──────────────┼─────────────────┼──────────────────────────┤
│ INV-REPLAY-HT-2 │ CRITICAL     │ Mark FAILED     │ Verify HT-1 status       │
│                 │              │ Block claims    │ Investigate if HT-1 pass │
│                 │              │ Preserve logs   │ Re-run experiment        │
├─────────────────┼──────────────┼─────────────────┼──────────────────────────┤
│ INV-REPLAY-HT-3 │ CRITICAL     │ Mark FAILED     │ Verify manifest match    │
│                 │              │ Block claims    │ Check MDAP consistency   │
│                 │              │ Preserve logs   │ Re-run experiment        │
├─────────────────┼──────────────┼─────────────────┼──────────────────────────┤
│ INV-REPLAY-HT-4 │ HIGH         │ Mark REVIEW     │ Compare receipt vs files │
│                 │              │ Flag for audit  │ Regenerate if needed     │
│                 │              │ Preserve all    │ Approve or reject        │
├─────────────────┼──────────────┼─────────────────┼──────────────────────────┤
│ INV-REPLAY-HT-5 │ HIGH         │ Mark REVIEW     │ Recompute binding        │
│                 │              │ Flag for audit  │ Fix computation if bug   │
│                 │              │ Preserve all    │ Approve or reject        │
└─────────────────┴──────────────┴─────────────────┴──────────────────────────┘
```

#### 10.5.4 Failure Logging Requirements

All replay failures MUST be logged with:

```json
{
  "failure_log": {
    "timestamp_utc": "<ISO8601>",
    "experiment_id": "<experiment_id>",
    "replay_id": "<replay_id>",
    "invariant_failed": "INV-REPLAY-HT-N",
    "severity": "<CRITICAL|HIGH>",
    "impact": "<INVALID|UNDER_REVIEW>",
    "details": {
      "expected": "<expected_value>",
      "actual": "<actual_value>",
      "divergence_cycle": null,
      "additional_context": {}
    },
    "artifacts_preserved": [
      "primary_ht_series.json",
      "replay_ht_series.json",
      "replay_receipt.json",
      "manifest.json"
    ],
    "auto_actions_taken": [
      "Marked experiment FAILED",
      "Blocked validity claims",
      "Notified governance"
    ],
    "human_actions_required": [
      "Root cause analysis",
      "Remediation",
      "Re-run decision"
    ]
  }
}
```

### 10.6 Domain Prefixes for Replay Binding

| Domain | Prefix | Purpose |
|--------|--------|---------|
| Primary-Replay Binding | `b"MathLedger:PrimaryReplayBinding:v2:"` | Links primary and replay chain finals |
| Replay Receipt | `b"MathLedger:ReplayReceipt:v2:"` | Receipt hash computation |

### 10.7 Replay-Hₜ Governance Obligations

#### GOV-REPLAY-1: Replay Timing

```
GOV-REPLAY-1: Replay MUST be executed within 30 days of primary run completion.
```

Rationale: Ensures environment consistency and artifact availability.

#### GOV-REPLAY-2: Replay Independence

```
GOV-REPLAY-2: Replay MUST be executable by an independent party.
```

Requirements:
- All artifacts needed for replay must be published
- No secret state required beyond MDAP seed (which is public)
- Replay instructions must be documented

#### GOV-REPLAY-3: Replay Attestation

```
GOV-REPLAY-3: Successful replay MUST produce a valid Replay Receipt.
```

The receipt serves as cryptographic proof that:
- Replay was performed
- Results matched (or documented divergence)
- Binding hashes are correct

---

## 11. Phase III Cryptographic Extensions

> **STATUS: FUTURE DESIGN — NOT PART OF PHASE II**
>
> This section describes potential cryptographic enhancements for Phase III.
> These are design considerations only and do not affect Phase II operations.

### 11.1 Planned Extensions

#### 11.1.1 Zero-Knowledge Proof of Schedule Compliance

**Motivation**: Allow verification that the Hₜ series was produced under the correct MDAP schedule without revealing the schedule itself.

**Approach**:
```
ZK-SCHEDULE := ZKProof {
    Public inputs:
        - mdap_attestation_hash
        - ht_chain_final
        - ht_mdap_binding

    Private inputs:
        - All cycle seeds
        - All R_t values

    Statement:
        "I know cycle_seeds and R_t values such that:
         1. cycle_seeds derive from MDAP_SEED via the specified formula
         2. H_t values are correctly computed from R_t and cycle_seeds
         3. ht_chain_final is the correct chain over H_t values
         4. ht_mdap_binding is correct"
}
```

**Benefits**:
- Provers cannot learn individual R_t values
- Verifiers can still confirm schedule compliance
- Supports audit without full data disclosure

#### 11.1.2 Threshold Signatures for Governance

**Motivation**: Require multiple parties to sign off on experiment validity.

**Approach**:
```
THRESHOLD-SIG := ThresholdSignature(t, n) {
    Signers: [governance_key_1, ..., governance_key_n]
    Threshold: t (require t of n signatures)

    Message: manifest_hash || verification_report_hash

    Signature: Combined signature valid iff >= t signers participated
}
```

**Benefits**:
- No single party can unilaterally approve experiments
- Resistant to key compromise (up to n-t keys)
- Supports distributed governance

#### 11.1.3 Verifiable Delay Functions for Timestamps

**Motivation**: Prove that a timestamp is accurate (not backdated).

**Approach**:
```
VDF-TIMESTAMP := {
    input: current_time || random_beacon
    output: VDF(input, difficulty)
    proof: VDF_proof

    Verification:
        1. Verify VDF_proof is valid for output
        2. VDF computation takes ~T seconds
        3. Therefore timestamp is within T seconds of actual time
}
```

**Benefits**:
- Prevents backdating MDAP attestation
- Does not require trusted timestamping authority
- Publicly verifiable

#### 11.1.4 Merkle Mountain Range for Incremental Proofs

**Motivation**: Enable efficient proofs of individual cycles without revealing entire series.

**Approach**:
```
MMR-SERIES := MerkleMountainRange {
    Append-only structure

    Operations:
        - append(H_t) → mmr_root_new
        - prove_inclusion(cycle, H_t) → proof
        - verify_inclusion(mmr_root, cycle, H_t, proof) → bool

    Properties:
        - O(log n) proof size
        - O(log n) verification time
        - Append-only (no modification)
}
```

**Benefits**:
- Efficient proofs for individual cycles
- Supports streaming verification
- Compatible with existing Hₜ chain

### 11.2 Migration Considerations

#### 11.2.1 Backward Compatibility

Phase III extensions MUST:
- Not invalidate Phase II artifacts
- Support verification of Phase II series without new crypto
- Use new domain prefixes (`:v3:`)

#### 11.2.2 Upgrade Path

```
Phase II artifacts:
    - Continue to use v2 domains
    - Remain valid and verifiable
    - No re-attestation required

Phase III artifacts:
    - Use v3 domains
    - Include Phase II-compatible fallback
    - New crypto is optional enhancement

Cross-phase verification:
    - Verifier detects version from domain prefix
    - Applies appropriate verification algorithm
    - Reports version in verification report
```

### 11.3 Research Dependencies

| Extension | Research Status | Dependencies |
|-----------|-----------------|--------------|
| ZK Schedule Proof | Theoretical | zkSNARK/zkSTARK library selection |
| Threshold Signatures | Mature | Threshold ECDSA or BLS library |
| VDF Timestamps | Experimental | VDF implementation (e.g., Chia VDF) |
| MMR Series | Mature | MMR library (e.g., from Grin/Mimblewimble) |

### 11.4 Phase III Timeline (Tentative)

```
Phase III-α: Research and Design
    - Evaluate ZK proof systems
    - Design threshold signature scheme
    - Prototype VDF integration

Phase III-β: Implementation
    - Implement selected extensions
    - Backward compatibility testing
    - Security audit

Phase III-γ: Deployment
    - Gradual rollout with Phase II fallback
    - Monitoring and adjustment
    - Full transition
```

---

## 12. Appendices

### Appendix A: Domain Prefix Reference

All domain prefixes used in Hₜ series governance:

| Domain | Prefix | Version | Purpose |
|--------|--------|---------|---------|
| Cycle Seed | `b"MathLedger:CycleSeed:v2:"` | 2 | Per-cycle seed derivation |
| Schedule | `b"MathLedger:SeedSchedule:v2:"` | 2 | Schedule hash |
| MDAP Attestation | `b"MathLedger:MDAPAttestation:v2:"` | 2 | Pre-execution attestation |
| Hₜ Cycle | `b"MathLedger:HtCycle:v2:"` | 2 | Per-cycle attestation |
| Hₜ Chain | `b"MathLedger:HtChain:v2:"` | 2 | Cumulative chain |
| Hₜ-MDAP Binding | `b"MathLedger:HtMdapBinding:v2:"` | 2 | Chain-attestation binding |
| Prereg Binding | `b"MathLedger:PreregBinding:v2:"` | 2 | Manifest-prereg binding |
| Success Metric | `b"MathLedger:SuccessMetric:v2:"` | 2 | Metric parameter hash |
| Primary-Replay Binding | `b"MathLedger:PrimaryReplayBinding:v2:"` | 2 | Links primary and replay chains |
| Replay Receipt | `b"MathLedger:ReplayReceipt:v2:"` | 2 | Receipt hash computation |

### Appendix B: Governance Obligation Summary

| ID | Obligation | Phase | Enforcement |
|----|------------|-------|-------------|
| GOV-PRE-1 | Preregistration complete | Pre-exec | Automated gate |
| GOV-PRE-2 | MDAP attestation recorded | Pre-exec | Automated gate |
| GOV-PRE-3 | Manifest initialized | Pre-exec | Automated gate |
| GOV-EXEC-1 | Determinism from seeds only | During | Per-cycle check |
| GOV-EXEC-2 | All fields recorded per cycle | During | Per-cycle check |
| GOV-EXEC-3 | No retroactive modification | During | Chain structure |
| GOV-POST-1 | Series finalized promptly | Post-exec | Timestamp check |
| GOV-POST-2 | Manifest completed | Post-exec | Verification |
| GOV-POST-3 | Verification passes | Post-exec | Automated gate |
| GOV-POST-4 | Artifacts preserved | Post-exec | Audit |
| GOV-REPLAY-1 | Replay within 30 days | Replay | Timestamp check |
| GOV-REPLAY-2 | Replay independently executable | Replay | Documentation |
| GOV-REPLAY-3 | Valid replay receipt produced | Replay | Verification |

### Appendix C: Non-Manipulability Constraint Summary

| ID | Constraint | Enforcement |
|----|------------|-------------|
| MANIP-PRE-1 | Target hash pre-commitment | Prereg binding |
| MANIP-PRE-2 | Success criteria pre-commitment | Success metric hash |
| MANIP-PRE-3 | Cycle count pre-commitment | MDAP attestation |
| MANIP-EXEC-1 | No selective recording | Chain integrity |
| MANIP-EXEC-2 | No outcome-dependent termination | Cycle count check |
| MANIP-EXEC-3 | No parallel experiments | Governance tracking |
| MANIP-POST-1 | Analysis method fixed | Prereg binding |
| MANIP-POST-2 | Full results reported | Audit |
| MANIP-POST-3 | No data modification | File hashes |

### Appendix D: Invariant Cross-Reference

| Charter Section | Related Invariants |
|-----------------|-------------------|
| §3 Governance Obligations | INV-PREREG-*, INV-MANIFEST-* |
| §4 Attestation Sequencing | INV-MDAP-ATTEST-*, INV-SCHEDULE-* |
| §5 MDAP Binding | INV-MDAP-ATTEST-*, INV-HT-CRYPTO-3, INV-HT-CRYPTO-4 |
| §6 Non-Manipulability | INV-PREREG-2, INV-HT-CHAIN-1, INV-HTSERIES-1 |
| §10 Replay-Hₜ Binding | INV-REPLAY-HT-1, INV-REPLAY-HT-2, INV-REPLAY-HT-3, INV-REPLAY-HT-4, INV-REPLAY-HT-5 |

### Appendix E: Glossary

| Term | Definition |
|------|------------|
| BE32 | 32-bit big-endian byte encoding |
| Chain | Cumulative hash over Hₜ series |
| Checkpoint | Periodic snapshot of chain state |
| Cycle | Single iteration of derivation experiment |
| Domain separation | Prefix distinguishing hash contexts |
| Gate | Automated check that must pass to proceed |
| MDAP | MathLedger Deterministic Attestation Protocol |
| Preregistration | Pre-committed experiment parameters |
| Primary run | Original experiment execution |
| Replay | Re-execution of experiment to verify determinism |
| Replay Receipt | Cryptographic proof of replay verification |
| Rₜ | Proof Merkle root for cycle t |
| Hₜ | Attestation hash for cycle t |
| Triangle | MDAP-Hₜ-Replay binding structure |
| UTF8 | UTF-8 string encoding |
| VCS | Version control system |

---

## Document Signature

```
Charter: H_T_SERIES_GOVERNANCE_CHARTER.md
Version: 1.0.0
Hash: <to be computed upon approval>

Status: DRAFT — AWAITING GOVERNANCE APPROVAL

This document becomes binding upon:
1. Governance Authority approval
2. Hash commitment to version control
3. Publication to experiment executors
```

---

*End of H_T_SERIES_GOVERNANCE_CHARTER.md*
