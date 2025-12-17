# Last-Mile Governance Specification — CLAUDE K

**Document Version:** 1.0.0
**Status:** DESIGN SPECIFICATION
**Phase:** X (Final Pass Engine)
**Classification:** STRATCOM — CLAUDE K
**Date:** 2025-12-10

---

## 1. Executive Summary

The Last-Mile Governance Checker is the final gating layer before any governance decision, action, or artifact is committed. It enforces a comprehensive final-pass validation that aggregates signals from all upstream layers (USLA, TDA, CDI, Invariants, Safe Region) into a single authoritative go/no-go decision.

### Design Principle

```
LAST-MILE CONTRACT

All governance decisions must pass through final-check validation.
No bypass paths exist. No exceptions without explicit audit trail.
The final gate is the sole authority for commitment.
```

---

## 2. Architectural Position

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GOVERNANCE PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   USLA      │  │    TDA      │  │    CDI      │  │  Invariant  │        │
│  │  Simulator  │  │  Monitor    │  │  Detector   │  │   Monitor   │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │                │
│         ▼                ▼                ▼                ▼                │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                    SIGNAL AGGREGATOR                              │      │
│  │                                                                   │      │
│  │  Collects:                                                        │      │
│  │  - USLA state vector (H, ρ, τ, β, Γ, J, C, W)                    │      │
│  │  - TDA metrics (SNS, PCS, DRS, HSS)                              │      │
│  │  - CDI activations (δ, active CDIs)                              │      │
│  │  - Invariant status (I₁...I₈)                                    │      │
│  │  - Safe region membership (∈ Ω)                                  │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                │                                            │
│                                ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │              ╔═══════════════════════════════════════╗            │      │
│  │              ║     LAST-MILE GOVERNANCE CHECKER      ║            │      │
│  │              ║           (CLAUDE K)                   ║            │      │
│  │              ╚═══════════════════════════════════════╝            │      │
│  │                                                                   │      │
│  │  Final Pass Validation:                                          │      │
│  │  ┌─────────────────────────────────────────────────────────┐    │      │
│  │  │ 1. Hard Gate Check         → GATE_HARD                  │    │      │
│  │  │ 2. Soft Gate Evaluation    → GATE_SOFT                  │    │      │
│  │  │ 3. Override Audit          → GATE_OVERRIDE              │    │      │
│  │  │ 4. Commitment Authority    → FINAL_VERDICT              │    │      │
│  │  └─────────────────────────────────────────────────────────┘    │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                │                                            │
│                                ▼                                            │
│                    ┌─────────────────────┐                                 │
│                    │   COMMITMENT GATE   │                                 │
│                    │   ALLOW | BLOCK     │                                 │
│                    └─────────────────────┘                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Final Gate Logic

### 3.1 Gate Hierarchy

The Last-Mile Checker evaluates gates in strict order. Earlier gates have precedence.

| Gate Level | Name | Condition | Override Possible |
|------------|------|-----------|-------------------|
| G0 | **GATE_CATASTROPHIC** | CDI-010 active | NO |
| G1 | **GATE_HARD** | HARD_OK = False for > N cycles | NO |
| G2 | **GATE_INVARIANT** | Any INV violated | Requires explicit waiver |
| G3 | **GATE_SAFE_REGION** | x ∉ Ω for > M cycles | Requires explicit waiver |
| G4 | **GATE_SOFT** | ρ < ρ_min OR β > β_max | YES (with audit) |
| G5 | **GATE_ADVISORY** | TDA metrics degraded | YES (informational) |

### 3.2 Gate Evaluation Rules

```
FINAL_VERDICT = ALLOW if and only if:
    (G0 = PASS) ∧
    (G1 = PASS) ∧
    (G2 = PASS ∨ G2_waiver) ∧
    (G3 = PASS ∨ G3_waiver) ∧
    (G4 = PASS ∨ G4_override)

FINAL_VERDICT = BLOCK otherwise.

G5 is advisory-only and does not affect FINAL_VERDICT.
```

### 3.3 Detailed Gate Specifications

#### G0: Catastrophic Gate

| Field | Specification |
|-------|---------------|
| **Trigger** | CDI-010 (Fixed-Point Multiplicity) activated |
| **Threshold** | Any single activation |
| **Severity** | CRITICAL |
| **Action** | Immediate BLOCK, no override |
| **Rationale** | Multiple fixed points indicate non-deterministic governance; unsafe to proceed |

#### G1: Hard Gate

| Field | Specification |
|-------|---------------|
| **Trigger** | HARD_OK = False |
| **Threshold** | Consecutive cycles > `hard_fail_threshold` (default: 50) |
| **Severity** | CRITICAL |
| **Action** | BLOCK, no override |
| **Rationale** | Sustained HARD mode failure indicates systemic instability |

#### G2: Invariant Gate

| Field | Specification |
|-------|---------------|
| **Trigger** | Any invariant INV-001 through INV-008 violated |
| **Threshold** | violation_count > `invariant_tolerance` (default: 0) |
| **Severity** | HIGH |
| **Action** | BLOCK, waiver possible |
| **Waiver Condition** | Explicit `invariant_waiver` with justification and expiry |

#### G3: Safe Region Gate

| Field | Specification |
|-------|---------------|
| **Trigger** | State x outside safe region Ω |
| **Threshold** | Consecutive cycles > `omega_exit_threshold` (default: 100) |
| **Severity** | HIGH |
| **Action** | BLOCK, waiver possible |
| **Waiver Condition** | Explicit `omega_waiver` with boundary analysis |

#### G4: Soft Gate

| Field | Specification |
|-------|---------------|
| **Trigger** | (ρ < ρ_min) OR (β > β_max) |
| **Thresholds** | `rho_min`: 0.4, `beta_max`: 0.6 |
| **Streak Required** | 10 cycles for ρ, 20 cycles for β |
| **Severity** | MEDIUM |
| **Action** | BLOCK, override possible |
| **Override Condition** | `soft_override` flag with audit trail |

#### G5: Advisory Gate

| Field | Specification |
|-------|---------------|
| **Trigger** | TDA metric degradation (SNS, PCS, DRS, or HSS below thresholds) |
| **Thresholds** | Configurable per metric |
| **Severity** | LOW (informational) |
| **Action** | LOG, no block |
| **Purpose** | Early warning for trending instability |

---

## 4. Input Contract

### 4.1 Required Signals

```python
@dataclass
class GovernanceFinalCheckInput:
    """Input signals for Last-Mile Governance Check."""

    # Cycle identification
    cycle: int
    timestamp: str  # ISO 8601

    # USLA State (from USLASimulator)
    usla_state: USLAStateVector

    # HARD mode status
    hard_ok: bool
    hard_fail_streak: int

    # Safe region status
    in_omega: bool
    omega_exit_streak: int

    # Invariant status
    invariant_violations: List[str]  # e.g., ["INV-001", "INV-003"]
    invariant_all_pass: bool

    # CDI status
    active_cdis: List[str]  # e.g., ["CDI-007"]
    cdi_010_active: bool

    # Stability metrics
    rho: float  # Rolling Stability Index
    rho_collapse_streak: int
    beta: float  # Block rate
    beta_explosion_streak: int

    # TDA metrics (optional, for advisory)
    tda_metrics: Optional[TDAMetrics]

    # Override/Waiver inputs
    waivers: List[GovernanceWaiver]
    overrides: List[GovernanceOverride]
```

### 4.2 USLAStateVector

```python
@dataclass
class USLAStateVector:
    """Canonical 15-element state vector per USLA v0.1."""
    H: float        # HSS [0, 1]
    D: int          # Depth
    D_dot: float    # Depth velocity
    B: float        # Branch factor
    S: float        # Shear [0, 1]
    C: int          # Convergence class {0=CONVERGING, 1=OSCILLATING, 2=DIVERGING}
    rho: float      # RSI [0, 1]
    tau: float      # Threshold [0.1, 0.5]
    J: float        # Jacobian sensitivity
    W: bool         # Exception window active
    beta: float     # Block rate [0, 1]
    kappa: float    # Coupling strength [0, 1]
    nu: float       # Variance velocity
    delta: int      # CDI defect count
    Gamma: float    # TGRS [0, 1]
```

### 4.3 TDAMetrics

```python
@dataclass
class TDAMetrics:
    """Topological Data Analysis metrics."""
    sns: float  # Structural Non-Triviality Score
    pcs: float  # Persistence Coherence Score
    drs: float  # Deviation-from-Reference Score
    hss: float  # Hallucination Stability Score (composite)
```

---

## 5. Output Contract

### 5.1 Final Check Result

```python
@dataclass
class GovernanceFinalCheckResult:
    """Result of Last-Mile Governance Check."""

    # Identification
    cycle: int
    timestamp: str
    check_version: str  # "1.0.0"

    # Final verdict
    verdict: Literal["ALLOW", "BLOCK"]
    verdict_confidence: float  # [0, 1]

    # Gate evaluations
    gates: GateEvaluations

    # Blocking gate (if verdict = BLOCK)
    blocking_gate: Optional[str]  # e.g., "G0_CATASTROPHIC"
    blocking_reason: Optional[str]

    # Active waivers/overrides applied
    waivers_applied: List[str]
    overrides_applied: List[str]

    # Audit hash
    audit_hash: str  # SHA-256 of input signals

    # Telemetry snapshot for replay
    input_snapshot_hash: str
```

### 5.2 GateEvaluations

```python
@dataclass
class GateEvaluations:
    """Individual gate evaluation results."""

    g0_catastrophic: GateResult
    g1_hard: GateResult
    g2_invariant: GateResult
    g3_safe_region: GateResult
    g4_soft: GateResult
    g5_advisory: GateResult  # Never blocks

@dataclass
class GateResult:
    """Single gate evaluation."""
    gate_id: str
    status: Literal["PASS", "FAIL", "WAIVED", "OVERRIDDEN"]
    trigger_value: Optional[float]
    threshold: Optional[float]
    streak: Optional[int]
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    details: Optional[str]
```

---

## 6. Waiver and Override System

### 6.1 Waiver Structure

Waivers provide temporary exceptions to hard gates (G2, G3) with explicit justification.

```python
@dataclass
class GovernanceWaiver:
    """Waiver for gate bypass with audit trail."""

    waiver_id: str
    gate_id: str  # "G2_INVARIANT" or "G3_SAFE_REGION"
    issued_by: str  # "human_operator" or "policy_engine"
    issued_at: str  # ISO 8601
    expires_at: str  # ISO 8601
    justification: str
    conditions: List[str]  # Conditions under which waiver remains valid
    max_cycles: int  # Maximum cycles waiver is valid
    signature: str  # Cryptographic signature
```

### 6.2 Override Structure

Overrides allow soft gate (G4) bypass with mandatory audit logging.

```python
@dataclass
class GovernanceOverride:
    """Override for soft gate with audit trail."""

    override_id: str
    gate_id: str  # "G4_SOFT"
    issued_by: str
    issued_at: str
    reason: str
    valid_for_cycles: int
    auto_revoke_conditions: List[str]
    audit_required: bool  # Always True
```

---

## 7. Audit Requirements

### 7.1 Immutable Audit Log

Every final check produces an immutable audit record.

| Field | Description |
|-------|-------------|
| `check_id` | UUID for this check |
| `cycle` | Cycle number |
| `timestamp` | ISO 8601 timestamp |
| `input_hash` | SHA-256 of all input signals |
| `verdict` | ALLOW or BLOCK |
| `gates_evaluated` | Array of gate results |
| `blocking_gate` | Which gate blocked (if any) |
| `waivers_applied` | Active waivers used |
| `overrides_applied` | Active overrides used |
| `output_hash` | SHA-256 of output |

### 7.2 Audit Properties

| Property | Guarantee |
|----------|-----------|
| **Immutability** | Append-only log, no modification |
| **Completeness** | Every check recorded, no gaps |
| **Verifiability** | Hash chain for integrity verification |
| **Traceability** | Full lineage from input to verdict |

---

## 8. Configuration Parameters

### 8.1 Gate Thresholds

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `hard_fail_threshold` | 50 | [10, 100] | Cycles before G1 blocks |
| `invariant_tolerance` | 0 | [0, 5] | Violations allowed before G2 |
| `omega_exit_threshold` | 100 | [20, 200] | Cycles outside Ω before G3 |
| `rho_min` | 0.4 | [0.2, 0.6] | RSI floor for G4 |
| `beta_max` | 0.6 | [0.4, 0.8] | Block rate ceiling for G4 |
| `rho_streak_threshold` | 10 | [5, 20] | Cycles of ρ < ρ_min for G4 |
| `beta_streak_threshold` | 20 | [10, 50] | Cycles of β > β_max for G4 |

### 8.2 TDA Advisory Thresholds

| Metric | Threshold | Description |
|--------|-----------|-------------|
| `sns_min` | 0.5 | SNS below triggers advisory |
| `pcs_min` | 0.6 | PCS below triggers advisory |
| `drs_max` | 0.3 | DRS above triggers advisory |
| `hss_min` | 0.5 | HSS below triggers advisory |

---

## 9. Implementation Mapping

| Component | Implementation File |
|-----------|-------------------|
| Last-Mile Checker | `backend/governance/last_mile_checker.py` |
| Gate Evaluator | `backend/governance/gate_evaluator.py` |
| Waiver Manager | `backend/governance/waiver_manager.py` |
| Override Handler | `backend/governance/override_handler.py` |
| Audit Logger | `backend/governance/audit_logger.py` |
| Schema Validation | `backend/governance/schemas.py` |

---

## 10. Integration Points

### 10.1 Upstream Dependencies

| System | Interface |
|--------|-----------|
| USLA Simulator | `USLAStateVector` via `USLABridge` |
| TDA Monitor | `TDAMetrics` via `TDAGovernanceHook` |
| CDI Detector | `active_cdis`, `cdi_010_active` |
| Invariant Monitor | `invariant_violations`, `invariant_all_pass` |
| Safe Region Checker | `in_omega`, `omega_exit_streak` |

### 10.2 Downstream Consumers

| Consumer | Interface |
|----------|-----------|
| Commitment Gate | `GovernanceFinalCheckResult.verdict` |
| Telemetry Logger | Full `GovernanceFinalCheckResult` |
| Dashboard | `gates`, `verdict`, `blocking_reason` |
| Audit Archive | Immutable audit record |

---

## 11. SHADOW MODE Considerations

In Phase X SHADOW mode:

| Behavior | Description |
|----------|-------------|
| **Compute All Gates** | All gates evaluated as in production |
| **Log Verdicts** | Full audit trail written |
| **No Enforcement** | Verdict is logged but not enforced |
| **Divergence Tracking** | Compare shadow verdict vs. real governance |

```python
# SHADOW MODE: Log but do not enforce
if config.shadow_mode:
    result = last_mile_check(input_signals)
    shadow_logger.log(result)
    # DO NOT USE result.verdict for control flow
```

---

## 12. Evidence Pack Binding (Phase Y)

### 12.1 Target Location

Upon exit from SHADOW MODE (Phase Y), governance final check artifacts will be integrated into the Evidence Pack at:

```
evidence["governance"]["final_check"]
```

**Status:** DESIGN SPECIFICATION ONLY — No inclusion in Evidence Pack until Phase Y.

### 12.2 Evidence Structure

```
evidence_pack_{run_id}/
└── governance/
    ├── final_check/
    │   ├── summary.json              # Aggregate check statistics
    │   ├── checks.jsonl              # All GovernanceFinalCheckResult records
    │   ├── blocks.jsonl              # BLOCK decisions only
    │   ├── waivers_applied.json      # Waiver usage summary
    │   └── overrides_applied.json    # Override usage summary
    ├── gate_stats/
    │   ├── g0_catastrophic.json
    │   ├── g1_hard.json
    │   ├── g2_invariant.json
    │   ├── g3_safe_region.json
    │   ├── g4_soft.json
    │   └── g5_advisory.json
    └── audit_chain/
        ├── chain_verification.json
        └── audit_log.jsonl
```

### 12.3 Evidence Binding Fields

| Field | Type | Description |
|-------|------|-------------|
| `summary.total_checks` | int | Total governance checks executed |
| `summary.allow_count` | int | Number of ALLOW verdicts |
| `summary.block_count` | int | Number of BLOCK verdicts |
| `summary.allow_rate` | float | ALLOW / total ratio |
| `gate_stats.{gate_id}.pass_rate` | float | Per-gate pass rate |
| `gate_stats.{gate_id}.blocking_count` | int | Times gate caused BLOCK |
| `audit_chain.chain_height` | int | Total records in chain |
| `audit_chain.verified` | bool | Chain integrity status |

### 12.4 Phase Y Activation Prerequisites

Evidence Pack inclusion requires:

1. SHADOW MODE divergence analysis complete
2. Gate calibration verified (no systematic false positives/negatives)
3. Audit chain integrity verified (hash chain valid)
4. Waiver/override policy framework ratified
5. **P5 empirical data available** — First smoke run completed and analyzed

See: [Evidence_Pack_Spec_PhaseX.md § 12](./Evidence_Pack_Spec_PhaseX.md#12-governance-final-check-integration-phase-y)

### 12.5 Current Status: HOLD POSITION

```
┌─────────────────────────────────────────────────────────────────┐
│  CLAUDE K — LAST-MILE GOVERNANCE CHECKER                        │
│  Status: HOLD POSITION                                          │
│                                                                 │
│  Awaiting P5 empirical data.                                    │
│  No Phase Y activation until first P5 smoke run is available.  │
│                                                                 │
│  Current posture:                                               │
│  • Gates G0–G5: DEFINED (spec complete)                        │
│  • Implementation: COMPLETE (47 tests passing)                  │
│  • Mode: SHADOW ONLY (observe, log, do not enforce)            │
│  • Evidence binding: SPEC ONLY (no pack inclusion yet)         │
│  • CI gating: DISABLED (advisory metrics only)                 │
│                                                                 │
│  Next action: Upon P5 smoke run availability, draft            │
│  hypothetical G3/G4 response analysis.                         │
└─────────────────────────────────────────────────────────────────┘
```

**Rationale:** Gate calibration requires real empirical data from P5 runs to validate thresholds. Premature activation risks false positives (blocking valid runs) or false negatives (passing unstable states). CLAUDE K remains in observation mode until data-driven calibration is possible.

### 12.6 Gate Alignment Panel (Pre-LastMile Reasoning)

The Gate Alignment Panel aggregates gate readiness annexes across CAL-EXP runs to provide calibration-level alignment analysis for auditors.

**Status:** SHADOW MODE ONLY — For pre-LastMile reasoning, stays SHADOW-ONLY until Phase Y gating is authorized.

**Location:**
```
evidence["governance"]["uplift_gate_alignment_panel"]
```

**Purpose:**
- Aggregate P3/P4 gate alignment across multiple calibration experiments
- Identify experiments where P3 and P4 safety gates disagree
- Provide alignment statistics for auditor review
- Track misalignment reason codes (P3_BLOCK, P4_BLOCK, BOTH_BLOCK)

**Panel Structure:**
```json
{
  "schema_version": "1.0.0",
  "total_experiments": 3,
  "aligned_count": 2,
  "misaligned_count": 1,
  "experiments_misaligned": ["CAL-EXP-2"],
  "alignment_rate": 0.667,
  "misalignment_details": [
    {
      "cal_id": "CAL-EXP-2",
      "reason_code": "P3_BLOCK",
      "p3_decision": "BLOCK",
      "p4_decision": "PASS"
    }
  ]
}
```

**Alignment Logic:**
- **Aligned**: P3 and P4 both PASS/WARN (or one PASS and one WARN)
- **Misaligned**: Either P3 or P4 is BLOCK
- **Reason Codes**: `P3_BLOCK` (only P3 is BLOCK), `P4_BLOCK` (only P4 is BLOCK), `BOTH_BLOCK` (both are BLOCK)

**Per-Experiment Export:**
Gate annexes are exported per CAL-EXP to:
```
calibration/uplift_gate_annex_<cal_id>.json
```

Each annex contains: `p3_decision`, `p3_risk_band`, `p4_decision`, `p4_risk_band`, `stability_trend`.

**Status Signal Integration:**
The panel is extracted (manifest-first) and attached to `first_light_status.json` under `signals["uplift_gate_alignment"]` with:
- `extraction_source`: `"MANIFEST"` | `"EVIDENCE_JSON"` | `"MISSING"` (provenance tracking)
- `top_reason_code`: Deterministically derived from `reason_code_histogram` (most frequent, tie-break by code)
- `alignment_rate`, `misaligned_count`, `top_misaligned_cal_ids`, `reason_code_histogram`

**GGFL Alignment View:**
The panel is normalized to GGFL format via `uplift_gate_alignment_for_alignment_view()`:
- `signal_type`: `"SIG-GATE"`
- `status`: `"ok"` | `"warn"` (warn if `misaligned_count > 0`)
- `conflict`: `false` (invariant)
- `drivers`: Reason-code drivers only (no prose):
  - `DRIVER_TOP_REASON_P3_BLOCK` | `DRIVER_TOP_REASON_P4_BLOCK` | `DRIVER_TOP_REASON_BOTH_BLOCK`
  - `DRIVER_MISALIGNED_COUNT_PRESENT`
  - `DRIVER_TOP_CAL_IDS_PRESENT`
- `shadow_mode_invariants`: `{advisory_only: true, no_enforcement: true, conflict_invariant: true}`
- `weight_hint`: `"LOW"` (advisory only, low weight)

**SHADOW MODE Contract:**
- Panel is purely observational
- No control flow depends on panel contents
- Does not influence LastMile gate decisions
- For pre-LastMile reasoning only
- Stays SHADOW-ONLY until Phase Y gating is authorized
- `conflict: false` is an invariant (gate alignment never conflicts directly)

---

## 13. CI Integration Design (Phase Y)

### 13.1 CI Job: `last_mile_governance_check`

**Status:** DESIGN STUB ONLY — No implementation until Phase Y.

#### 13.1.1 Job Specification

| Attribute | Specification |
|-----------|---------------|
| **Job Name** | `last_mile_governance_check` |
| **Trigger** | On completion of USLA shadow simulation |
| **Dependencies** | `usla_shadow_run`, `tda_metrics_compute` |
| **Runner** | Standard CI runner (no GPU required) |
| **Timeout** | 10 minutes |
| **Mode** | SHADOW (observe-only) |

#### 13.1.2 Execution Sequence

1. **Input Collection**
   - Load USLA state history from `usla_shadow_run` artifacts
   - Load TDA metrics from `tda_metrics_compute` artifacts
   - Load gate configuration from `config/governance_gates.yaml`

2. **Gate Evaluation**
   - Execute `run_governance_final_check()` for each cycle
   - Accumulate results in `GovernanceEvidencePack`

3. **Audit Chain Construction**
   - Build hash-linked audit log via `GovernanceAuditLogger`
   - Verify chain integrity

4. **Artifact Emission**
   - Write `governance/final_check/summary.json`
   - Write `governance/final_check/checks.jsonl`
   - Write `governance/gate_stats/*.json`
   - Write `governance/audit_chain/chain_verification.json`

#### 13.1.3 Artifact Outputs

| Artifact | Path | Description |
|----------|------|-------------|
| Summary | `results/governance/final_check/summary.json` | Aggregate statistics |
| Checks | `results/governance/final_check/checks.jsonl` | Full check records |
| Blocks | `results/governance/final_check/blocks.jsonl` | BLOCK decisions |
| Gate Stats | `results/governance/gate_stats/*.json` | Per-gate statistics |
| Chain Verification | `results/governance/audit_chain/chain_verification.json` | Integrity status |

#### 13.1.4 Success Criteria (Informational)

In SHADOW MODE, the job always succeeds. The following are **advisory** metrics logged but not gated:

| Metric | Advisory Threshold | Description |
|--------|-------------------|-------------|
| `allow_rate` | ≥ 0.95 | High allow rate indicates healthy system |
| `g0_catastrophic.fail_count` | = 0 | No CDI-010 activations |
| `g1_hard.fail_count` | < 5 | Few HARD mode failure streaks |
| `audit_chain.verified` | = true | Chain integrity maintained |

#### 13.1.5 Phase Y Gating (Future)

Upon Phase Y activation, the job will enforce:

```
GATE: allow_rate >= 0.90
GATE: g0_catastrophic.fail_count == 0
GATE: audit_chain.verified == true
```

**Note:** Enforcement gates are disabled in SHADOW MODE.

---

## 14. First Light Uplift Safety Gate Annex

### 14.1 Overview

The First Light Uplift Safety Gate Annex provides a compact summary of the state of the uplift safety gate for First Light evidence packs. It synthesizes P3 (synthetic) and P4 (shadow) safety gate decisions into a single alignment indicator.

**CRITICAL**: The annex is **NOT a gate** or enforcement primitive. It is an alignment indicator between P3 and P4 safety gates, not a hard enforcement primitive. The annex provides auditors with a quick view of whether P3 and P4 safety decisions are aligned, but does not influence any control flow or decisions.

### 14.2 Annex Structure

The annex is located at:
```
evidence["governance"]["uplift_safety"]["first_light_gate_annex"]
```

#### 14.2.1 Schema

```json
{
  "schema_version": "1.0.0",
  "p3_decision": "PASS" | "WARN" | "BLOCK",
  "p3_risk_band": "LOW" | "MEDIUM" | "HIGH",
  "p4_decision": "PASS" | "WARN" | "BLOCK",
  "p4_risk_band": "LOW" | "MEDIUM" | "HIGH",
  "stability_trend": "IMPROVING" | "STABLE" | "DEGRADING"
}
```

#### 14.2.2 Field Descriptions

- **`schema_version`**: Version identifier for the annex schema (currently "1.0.0").
- **`p3_decision`**: Uplift safety decision from P3 (synthetic) gate evaluation.
  - `PASS`: All safety indicators within acceptable ranges
  - `WARN`: Some safety indicators show degradation
  - `BLOCK`: Critical safety indicators indicate unsafe conditions
- **`p3_risk_band`**: Risk band from P3 safety tensor.
  - `LOW`: Low risk across all safety signals
  - `MEDIUM`: Moderate risk in some safety signals
  - `HIGH`: High risk across multiple safety signals
- **`p4_decision`**: Uplift safety decision from P4 (shadow) gate evaluation.
  - Same semantics as `p3_decision`
- **`p4_risk_band`**: Risk band from P4 safety tensor.
  - Same semantics as `p3_risk_band`
- **`stability_trend`**: Stability trend from P4 stability forecaster.
  - `IMPROVING`: Stability indicators improving over time
  - `STABLE`: Stability indicators stable
  - `DEGRADING`: Stability indicators degrading over time

### 14.3 Gate Alignment Indicator

The annex is used in conjunction with the `gate_alignment_ok` flag in the Uplift Council summary to indicate whether P3 and P4 safety gates are aligned.

#### 14.3.1 Alignment Semantics

The `gate_alignment_ok` flag (included in `summarize_uplift_safety_for_council()` output) indicates:

- **`gate_alignment_ok = True`**: P3 and P4 decisions are aligned
  - Both are `PASS`/`OK`, OR
  - Both are `WARN`, OR
  - One is `PASS`/`OK` and the other is `WARN`
- **`gate_alignment_ok = False`**: P3 and P4 decisions are not aligned
  - Either P3 or P4 is `BLOCK`

#### 14.3.2 Interpretation

**An alignment indicator between P3 and P4 safety gates, not a hard enforcement primitive.**

The `gate_alignment_ok` flag is intended as an auditor hint. If `False`, auditors should look harder at the discrepancy between P3 vs P4 safety decisions. The flag does not block or allow any operations; it is purely observational.

**Interpretation Patterns:**
- `gate_alignment_ok = True` + both `PASS`: Both gates agree on safe conditions
- `gate_alignment_ok = True` + both `WARN`: Both gates agree on moderate risk
- `gate_alignment_ok = True` + mixed `PASS`/`WARN`: Gates differ slightly but both indicate acceptable conditions
- `gate_alignment_ok = False` + one `BLOCK`: Gates disagree significantly; investigate discrepancy

### 14.4 Integration

The annex is automatically included in evidence packs when both P3 summary and P4 calibration data are available:

```python
from backend.health.uplift_safety_adapter import (
    attach_uplift_safety_to_evidence,
    build_p3_uplift_safety_summary,
    build_p4_uplift_safety_calibration,
)

# Build P3 and P4 summaries
p3_summary = build_p3_uplift_safety_summary(signal)
p4_calibration = build_p4_uplift_safety_calibration(tile)

# Attach to evidence (annex included automatically)
evidence = attach_uplift_safety_to_evidence(
    evidence, tile, signal, p3_summary, p4_calibration
)
# evidence["governance"]["uplift_safety"]["first_light_gate_annex"] contains annex
```

### 14.5 SHADOW MODE Contract

- The annex is purely observational
- No control flow depends on the annex contents
- The annex does not influence any governance decisions
- The annex is SHADOW MODE only, no gating
- All escalation decisions remain in upstream governance systems

---

## 15. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-10 | Initial Last-Mile Governance Specification |
| 1.1.0 | 2025-12-11 | Added § 12 Evidence Pack Binding, § 13 CI Integration Design |
| 1.1.1 | 2025-12-11 | Added § 12.5 HOLD POSITION status — awaiting P5 empirical data |
| 1.1.2 | 2025-12-11 | Added § 14 First Light Uplift Safety Gate Annex — alignment indicator documentation |
| 1.1.3 | 2025-12-11 | Added § 12.6 Gate Alignment Panel — pre-LastMile reasoning for CAL-EXP aggregation |
| 1.1.4 | 2025-12-11 | Updated § 12.6 Gate Alignment Panel — added status signal integration, GGFL adapter (SIG-GATE), unified invariants schema |

---

## 15. References

- [USLA v0.1](./USLA_v0.1.md) — Unified System Law Abstraction
- [Phase X Integration Spec](./Phase_X_Integration_Spec_v1.0.md) — Phase X system integration
- [Phase X P3 Spec](./Phase_X_P3_Spec.md) — First-Light shadow experiment
- [Phase X P4 Spec](./Phase_X_P4_Spec.md) — Real-runner shadow coupling
- [Evidence Pack Spec](./Evidence_Pack_Spec_PhaseX.md) — Evidence Pack structure and artifacts
