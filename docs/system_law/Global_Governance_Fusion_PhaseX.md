# Global Governance Fusion Layer — Phase X Specification

**Document Version:** 1.0.0
**Status:** Specification
**Parent:** USLA v0.1
**Phase:** X (SHADOW MODE ONLY)
**Date:** 2025-12-10

---

## 1. Purpose

The Global Governance Fusion Layer (GGFL) defines the unified interface for merging heterogeneous governance signals into a single, coherent governance decision. This specification establishes:

1. The complete taxonomy of input signals
2. Merging rules and conflict resolution
3. Precedence hierarchy for conflicting signals
4. Escalation model for unresolvable conflicts

**SHADOW MODE CONTRACT**: During Phase X, the GGFL operates in observational mode only. Fusion outputs are logged for analysis but do not influence actual governance decisions.

---

## 2. Input Signal Taxonomy

The GGFL receives signals from eight distinct subsystems. Each signal conforms to a defined schema and carries semantic meaning within the governance context.

### 2.1 Signal Inventory

| Signal ID | Name | Source Module | Update Frequency | Cardinality |
|-----------|------|---------------|------------------|-------------|
| `SIG-TOP` | Topology Signal | `backend/topology/` | Per-cycle | 1:1 |
| `SIG-RPL` | Replay Signal | `backend/replay/` | Per-cycle | 1:1 |
| `SIG-MET` | Metrics Signal | `backend/analytics/` | Per-cycle | 1:1 |
| `SIG-BUD` | Budget Signal | `backend/budget/` | Per-cycle | 1:1 |
| `SIG-STR` | Structure Signal | `backend/dag/` | Per-cycle | 1:1 |
| `SIG-TEL` | Telemetry Signal | `backend/telemetry/` | Per-cycle | 1:1 |
| `SIG-IDN` | Identity Signal | `substrate/crypto/` | Per-block | 1:1 |
| `SIG-NAR` | Narrative Signal | `backend/narrative/` | Per-epoch | 1:1 |
| `SIG-CON` | Consensus Signal | `backend/health/consensus_polygraph_adapter` | Per-cycle | 1:1 |

### 2.2 Signal Definitions

#### 2.2.1 Topology Signal (SIG-TOP)

**Description**: State vector from USLA simulator capturing topological health.

**Schema Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `H` | float [0,1] | Homological Stability Score |
| `D` | int | Current proof depth |
| `D_dot` | float | Depth velocity |
| `B` | float | Branch factor |
| `S` | float [0,1] | Semantic shear |
| `C` | enum {0,1,2} | Convergence class (CONVERGING, OSCILLATING, DIVERGING) |
| `rho` | float [0,1] | Rolling Stability Index |
| `tau` | float | Effective threshold |
| `J` | float | Jacobian sensitivity |
| `within_omega` | bool | Safe region membership |
| `active_cdis` | array[string] | Active CDI codes |
| `invariant_violations` | array[string] | Violated invariant IDs |

**Governance Semantics**:
- `within_omega = false` → BLOCK recommendation
- `C = 2` (DIVERGING) → BLOCK recommendation
- `rho < 0.4` → BLOCK recommendation

---

#### 2.2.2 Replay Signal (SIG-RPL)

**Description**: Safety assessment from replay verification system.

**Schema Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `replay_verified` | bool | Last proof successfully replayed |
| `replay_divergence` | float [0,1] | Divergence from canonical trace |
| `replay_latency_ms` | int | Replay verification latency |
| `replay_hash_match` | bool | Hash matches expected |
| `replay_depth_valid` | bool | Depth within expected bounds |

**Governance Semantics**:
- `replay_verified = false` → BLOCK recommendation
- `replay_divergence > 0.1` → WARNING, potential BLOCK
- `replay_hash_match = false` → HARD BLOCK (security critical)

---

#### 2.2.3 Metrics Signal (SIG-MET)

**Description**: Quantitative performance metrics from analytics subsystem.

**Schema Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `success_rate` | float [0,1] | Rolling success rate |
| `abstention_rate` | float [0,1] | Rolling abstention rate |
| `block_rate` | float [0,1] | Rolling block rate |
| `throughput` | float | Proofs per cycle |
| `latency_p50_ms` | int | Median verification latency |
| `latency_p99_ms` | int | 99th percentile latency |
| `queue_depth` | int | Pending verification queue |

**Governance Semantics**:
- `block_rate > 0.5` → EASE threshold recommendation
- `abstention_rate > 0.3` → WARNING
- `queue_depth > 1000` → THROTTLE recommendation

---

#### 2.2.4 Budget Signal (SIG-BUD)

**Description**: Resource budget constraints and utilization.

**Schema Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `compute_budget_remaining` | float [0,1] | Remaining compute allocation |
| `memory_utilization` | float [0,1] | Current memory usage |
| `storage_headroom_gb` | float | Available storage |
| `verification_quota_remaining` | int | Remaining verifications this epoch |
| `budget_exhaustion_eta_cycles` | int | Cycles until budget exhaustion |

**Governance Semantics**:
- `compute_budget_remaining < 0.1` → THROTTLE recommendation
- `verification_quota_remaining = 0` → HARD BLOCK
- `budget_exhaustion_eta_cycles < 10` → WARNING

---

#### 2.2.5 Structure Signal (SIG-STR)

**Description**: DAG structural health from proof graph analysis.

**Schema Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `dag_coherent` | bool | DAG passes consistency checks |
| `orphan_count` | int | Unlinked proof nodes |
| `max_fanout` | int | Maximum node children |
| `depth_distribution` | object | Histogram of proof depths |
| `cycle_detected` | bool | Cyclic dependency detected |
| `min_cut_capacity` | float | Minimum cut capacity |

**Governance Semantics**:
- `dag_coherent = false` → HARD BLOCK
- `cycle_detected = true` → HARD BLOCK (invariant violation)
- `orphan_count > 100` → WARNING
- `min_cut_capacity < 0.1` → BLOCK recommendation

---

#### 2.2.6 Telemetry Signal (SIG-TEL)

**Description**: Operational telemetry from runtime monitoring.

**Schema Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `lean_healthy` | bool | Lean verifier responsive |
| `db_healthy` | bool | Database connection healthy |
| `redis_healthy` | bool | Redis connection healthy |
| `worker_count` | int | Active worker processes |
| `error_rate` | float [0,1] | Recent error rate |
| `last_error` | string | Most recent error message |
| `uptime_seconds` | int | System uptime |

**Governance Semantics**:
- `lean_healthy = false` → HARD BLOCK
- `db_healthy = false` → HARD BLOCK
- `error_rate > 0.1` → WARNING, potential BLOCK
- `worker_count = 0` → HARD BLOCK

---

#### 2.2.7 Identity Signal (SIG-IDN)

**Description**: Cryptographic identity and provenance verification.

**Schema Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `block_hash_valid` | bool | Current block hash verified |
| `merkle_root_valid` | bool | Merkle root consistency |
| `signature_valid` | bool | Block signature verified |
| `chain_continuous` | bool | No gaps in block sequence |
| `pq_attestation_valid` | bool | Post-quantum attestation valid |
| `dual_root_consistent` | bool | Dual-root attestation consistent |

**Governance Semantics**:
- `block_hash_valid = false` → HARD BLOCK (security critical)
- `chain_continuous = false` → HARD BLOCK
- `pq_attestation_valid = false` → WARNING (PQ not mandatory yet)
- `dual_root_consistent = false` → HARD BLOCK

---

#### 2.2.8 Narrative Signal (SIG-NAR)

**Description**: High-level curriculum and strategic state.

**Schema Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `current_slice` | string | Active curriculum slice |
| `slice_progress` | float [0,1] | Progress through current slice |
| `epoch` | int | Current epoch number |
| `curriculum_health` | enum | HEALTHY, DEGRADED, CRITICAL |
| `drift_detected` | bool | Curriculum drift detected |
| `narrative_coherence` | float [0,1] | Narrative consistency score |

**Governance Semantics**:
- `curriculum_health = CRITICAL` → PAUSE recommendation
- `drift_detected = true` → WARNING
- `narrative_coherence < 0.5` → WARNING

---

#### 2.2.9 Consensus Signal (SIG-CON)

**Description**: Cross-layer disagreement index from consensus polygraph analysis.

**Schema Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `consensus_band` | enum | `HIGH`, `MEDIUM`, `LOW` — Overall consensus level |
| `agreement_rate` | float [0,1] | Proportion of slices where all systems agree |
| `conflict_count` | int | Number of cross-layer disagreements detected |
| `predictive_risk_band` | enum | `HIGH`, `MEDIUM`, `LOW`, `UNKNOWN` — Predictive conflict risk |

**First Light Conflict Ledger**:

The consensus signal includes a `first_light_conflict_ledger` annex that encodes the cross-layer disagreement index for First Light experiments. This ledger provides a compact summary of consensus health:

```json
{
  "schema_version": "1.0.0",
  "consensus_band": "MEDIUM",
  "agreement_rate": 0.75,
  "conflict_count": 3,
  "predictive_risk_band": "MEDIUM"
}
```

**Interpretation Guidelines for GGFL and Reviewers**:

The conflict ledger serves as a **disagreement index** that indicates the level of cross-system alignment:

- **Low conflict_count (0-2) + LOW/MEDIUM risk band**: "Consensus subsystem mostly stable." Systems are generally in agreement, with minimal cross-layer disagreements. This indicates healthy alignment across semantic, metric, drift, topology, and curriculum layers.

- **High conflict_count (>5) + HIGH risk band**: "Repeated cross-layer disagreement; treat as high-priority investigation area." Multiple systems disagree on slice/component statuses, and predictive analysis indicates elevated risk of further divergence. This warrants detailed investigation into the root causes of disagreement.

- **Medium conflict_count (3-5) + MEDIUM risk band**: "Moderate cross-layer divergence observed." Some systems disagree, but the disagreement is not yet systemic. Monitor for trends and investigate specific conflict cases.

**Governance Semantics**:
- `conflict_count > 5` AND `predictive_risk_band = HIGH` → BLOCK recommendation (priority 6)
- `conflict_count > 5` OR `predictive_risk_band = HIGH` → WARNING (priority 5)
- `consensus_band = LOW` → WARNING (priority 4)
- `agreement_rate < 0.5` → WARNING (priority 3)

**Note**: The conflict ledger is **observational only** in Phase X. It provides diagnostic information for fusion layer analysis and evidence pack review, but does not directly gate governance decisions. The ledger is designed as a compact disagreement summary for First Light evidence, not as a direct gating rule.

**Calibration Mode (CAL-EXP)**: For P5 calibration experiments (CAL-EXP-1/2/3), the consensus conflict register aggregates conflict ledgers across experiments to provide a summary view of cross-layer disagreement patterns. The register includes:
- Distribution of conflict_count buckets (0-2, 3-5, >5)
- Count of experiments with predictive_risk_band="HIGH"
- List of experiments exceeding high conflict threshold
- Consensus band and agreement rate distributions

The consensus conflict register is attached to evidence packs under `evidence["governance"]["consensus_conflict_register"]` and feeds into SIG-CON analysis in calibration mode. Like the conflict ledger, the register is **advisory only** and does not gate calibration decisions. It provides diagnostic context for understanding how often high conflict between subsystems occurs across calibration experiments.

**GGFL Alignment View Adapter**:

The SIG-CON adapter (`consensus_conflicts_for_alignment_view()`) produces normalized output for GGFL integration:

```json
{
  "signal_type": "SIG-CON",
  "status": "ok",
  "conflict": false,
  "drivers": ["DRIVER_FUSION_TENSION_OR_CONFLICT", "DRIVER_HIGH_CONFLICT_EXPERIMENTS_PRESENT"],
  "summary": "Consensus conflict register: fusion consistency status is TENSION (2 high-conflict, 1 high-risk experiments).",
  "extraction_source": "MANIFEST",
  "shadow_mode_invariants": {
    "advisory_only": true,
    "no_enforcement": true,
    "conflict_invariant": true
  }
}
```

**Driver Reason Codes (SIG-CON CONTRACT v1 FREEZE)**:
- `DRIVER_FUSION_TENSION_OR_CONFLICT`: Fusion consistency status is TENSION or CONFLICT
- `DRIVER_HIGH_CONFLICT_EXPERIMENTS_PRESENT`: At least one experiment exceeds high conflict threshold
- `DRIVER_HIGH_RISK_BAND_PRESENT`: At least one experiment has predictive_risk_band="HIGH"

**Driver Ordering**: Deterministic ordering: FUSION_TENSION_OR_CONFLICT → HIGH_CONFLICT_EXPERIMENTS → HIGH_RISK_BAND (max 3 drivers)

**Shadow Mode Invariants**:
- `advisory_only: true`: Signal is purely advisory, no enforcement
- `no_enforcement: true`: No systems enforce based on this signal
- `conflict_invariant: true`: Conflict field is always `false` (invariant)

**Status Determination**: Status is "warn" if `fusion_consistency_status` is TENSION/CONFLICT OR `experiments_high_conflict_count > 0` OR `high_risk_band_count > 0`; otherwise "ok".

---

## 3. Merging Rules

### 3.1 Signal Aggregation

All signals are aggregated into a unified governance envelope:

```
GovernanceEnvelope = {
    timestamp: ISO8601,
    cycle: int,
    signals: {
        topology: SIG-TOP,
        replay: SIG-RPL,
        metrics: SIG-MET,
        budget: SIG-BUD,
        structure: SIG-STR,
        telemetry: SIG-TEL,
        identity: SIG-IDN,
        narrative: SIG-NAR,
        consensus: SIG-CON
    },
    fusion_result: FusionResult
}
```

### 3.2 Signal Validation

Before merging, each signal undergoes validation:

1. **Schema Compliance**: Signal conforms to expected schema
2. **Freshness**: Signal timestamp within acceptable staleness window
3. **Completeness**: Required fields present
4. **Consistency**: Cross-signal consistency checks pass

**Staleness Windows**:
| Signal | Max Staleness |
|--------|---------------|
| SIG-TOP | 1 cycle |
| SIG-RPL | 1 cycle |
| SIG-MET | 1 cycle |
| SIG-BUD | 1 cycle |
| SIG-STR | 1 cycle |
| SIG-TEL | 1 cycle |
| SIG-IDN | 1 block |
| SIG-NAR | 1 epoch |

### 3.3 Recommendation Extraction

Each signal produces zero or more governance recommendations:

```
Recommendation = {
    signal_id: string,
    action: enum {ALLOW, BLOCK, HARD_BLOCK, THROTTLE, EASE, PAUSE, WARNING},
    confidence: float [0,1],
    reason: string,
    priority: int [1-10]
}
```

**Action Semantics**:
| Action | Meaning | Reversible |
|--------|---------|------------|
| ALLOW | Permit cycle to proceed | N/A |
| BLOCK | Soft block, subject to override | Yes |
| HARD_BLOCK | Mandatory block, no override | No |
| THROTTLE | Reduce throughput | Yes |
| EASE | Relax threshold | Yes |
| PAUSE | Pause curriculum advancement | Yes |
| WARNING | Log for monitoring | Yes |

### 3.4 Fusion Algorithm

```
FUNCTION fuse_signals(signals: Dict[str, Signal]) -> FusionResult:

    recommendations = []

    FOR signal_id, signal IN signals:
        IF NOT validate_signal(signal):
            recommendations.append(Recommendation(
                signal_id=signal_id,
                action=WARNING,
                confidence=1.0,
                reason="Signal validation failed",
                priority=5
            ))
            CONTINUE

        recommendations.extend(extract_recommendations(signal_id, signal))

    // Apply precedence rules
    hard_blocks = [r for r in recommendations if r.action == HARD_BLOCK]
    IF hard_blocks:
        RETURN FusionResult(
            decision=BLOCK,
            is_hard=True,
            primary_reason=hard_blocks[0].reason,
            recommendations=recommendations
        )

    blocks = [r for r in recommendations if r.action == BLOCK]
    allows = [r for r in recommendations if r.action == ALLOW]

    // Weighted voting for soft decisions
    block_score = sum(r.confidence * r.priority for r in blocks)
    allow_score = sum(r.confidence * r.priority for r in allows) + DEFAULT_ALLOW_BIAS

    IF block_score > allow_score:
        RETURN FusionResult(
            decision=BLOCK,
            is_hard=False,
            primary_reason=max(blocks, key=lambda r: r.confidence * r.priority).reason,
            recommendations=recommendations
        )
    ELSE:
        RETURN FusionResult(
            decision=ALLOW,
            is_hard=False,
            primary_reason="All signals nominal",
            recommendations=recommendations
        )
```

---

## 4. Conflict Precedence

### 4.1 Precedence Hierarchy

When signals conflict, the following precedence order applies (highest to lowest):

```
PRECEDENCE_ORDER = [
    1. Identity (SIG-IDN)      -- Cryptographic integrity is paramount
    2. Structure (SIG-STR)     -- DAG consistency required for correctness
    3. Telemetry (SIG-TEL)     -- System must be operational
    4. Replay (SIG-RPL)        -- Verification integrity
    5. Topology (SIG-TOP)      -- Stability assessment
    6. Budget (SIG-BUD)        -- Resource constraints
    7. Metrics (SIG-MET)       -- Performance indicators
    8. Consensus (SIG-CON)    -- Cross-layer alignment (advisory)
    9. Narrative (SIG-NAR)     -- Strategic guidance (advisory)
]
```

### 4.2 Conflict Resolution Matrix

| Conflict Type | Resolution Rule |
|---------------|-----------------|
| HARD_BLOCK vs. any | HARD_BLOCK wins unconditionally |
| BLOCK vs. ALLOW | Higher precedence signal wins |
| BLOCK vs. BLOCK | Union (both reasons logged) |
| ALLOW vs. ALLOW | Union (proceed) |
| THROTTLE vs. EASE | THROTTLE wins (conservative) |
| WARNING vs. any | Logged; does not affect decision |

### 4.3 Cross-Signal Consistency Rules

**Rule CSC-001**: If `SIG-IDN.chain_continuous = false` AND `SIG-STR.dag_coherent = true`, escalate as CONFLICT_DETECTED (possible data inconsistency).

**Rule CSC-002**: If `SIG-TOP.within_omega = true` AND `SIG-MET.block_rate > 0.5`, escalate as CONFLICT_DETECTED (metrics disagree with topology).

**Rule CSC-003**: If `SIG-TEL.lean_healthy = true` AND `SIG-RPL.replay_verified = false`, escalate as CONFLICT_DETECTED (verifier healthy but replay failed).

**Rule CSC-004**: If `SIG-BUD.verification_quota_remaining > 0` AND `SIG-TEL.worker_count = 0`, escalate as CONFLICT_DETECTED (budget available but no workers).

---

## 5. Escalation Model

### 5.1 Escalation Levels

| Level | Name | Trigger | Response |
|-------|------|---------|----------|
| L0 | NOMINAL | All signals healthy | Normal operation |
| L1 | WARNING | Any WARNING recommendation | Log and monitor |
| L2 | DEGRADED | Any BLOCK (soft) recommendation | Increased monitoring, potential BLOCK |
| L3 | CRITICAL | Any HARD_BLOCK recommendation | Immediate BLOCK, alert operators |
| L4 | CONFLICT | Cross-signal consistency failure | Escalate to human review |
| L5 | EMERGENCY | Multiple L3 conditions OR identity failure | Halt all operations, full audit |

### 5.2 Escalation Triggers

```
FUNCTION compute_escalation_level(fusion_result: FusionResult) -> EscalationLevel:

    hard_blocks = count(r for r in fusion_result.recommendations if r.action == HARD_BLOCK)
    soft_blocks = count(r for r in fusion_result.recommendations if r.action == BLOCK)
    warnings = count(r for r in fusion_result.recommendations if r.action == WARNING)
    conflicts = count(fusion_result.conflict_detections)

    // Identity or structure failure is always L5
    IF any(r.signal_id in ['identity', 'structure'] AND r.action == HARD_BLOCK):
        RETURN L5_EMERGENCY

    // Multiple hard blocks is L5
    IF hard_blocks >= 2:
        RETURN L5_EMERGENCY

    // Any cross-signal conflict
    IF conflicts > 0:
        RETURN L4_CONFLICT

    // Any hard block
    IF hard_blocks >= 1:
        RETURN L3_CRITICAL

    // Multiple soft blocks
    IF soft_blocks >= 2:
        RETURN L2_DEGRADED

    // Any warning
    IF warnings > 0:
        RETURN L1_WARNING

    RETURN L0_NOMINAL
```

### 5.3 Escalation Actions

| Level | Automated Actions | Manual Review Required |
|-------|-------------------|------------------------|
| L0 | None | No |
| L1 | Log to warning stream | No |
| L2 | Log, activate exception window if available | Optional |
| L3 | Log, BLOCK cycle, emit alert | Recommended |
| L4 | Log, BLOCK cycle, emit alert, flag for audit | Required |
| L5 | Log, HALT operations, emit critical alert | Required immediately |

### 5.4 Alert Channels

| Channel | Levels | Format |
|---------|--------|--------|
| Log file | L0-L5 | JSONL |
| Console | L1-L5 | Text |
| Webhook | L3-L5 | JSON payload |
| Email | L4-L5 | Summary digest |
| PagerDuty | L5 | Critical alert |

---

## 6. Observability Integration

### 6.1 Fusion Tile Schema

The fusion layer produces a health tile for the GlobalHealthSurface:

```json
{
    "tile_type": "governance_fusion",
    "schema_version": "1.0.0",
    "timestamp": "2025-12-10T12:00:00.000Z",
    "cycle": 1234,
    "escalation_level": "L0_NOMINAL",
    "decision": "ALLOW",
    "is_hard": false,
    "signal_summary": {
        "topology": {"status": "healthy", "recommendations": 0},
        "replay": {"status": "healthy", "recommendations": 0},
        "metrics": {"status": "healthy", "recommendations": 0},
        "budget": {"status": "healthy", "recommendations": 0},
        "structure": {"status": "healthy", "recommendations": 0},
        "telemetry": {"status": "healthy", "recommendations": 0},
        "identity": {"status": "healthy", "recommendations": 0},
        "narrative": {"status": "healthy", "recommendations": 0},
        "consensus": {"status": "healthy", "recommendations": 0}
    },
    "conflict_count": 0,
    "headline": "All signals nominal; governance fusion ALLOW"
}
```

### 6.2 Logging Schema

Per-cycle fusion logs are written in JSONL format:

```json
{
    "timestamp": "2025-12-10T12:00:00.000Z",
    "cycle": 1234,
    "signals_received": 8,
    "signals_valid": 8,
    "recommendations_total": 8,
    "recommendations_by_action": {
        "ALLOW": 8,
        "BLOCK": 0,
        "HARD_BLOCK": 0,
        "WARNING": 0
    },
    "fusion_decision": "ALLOW",
    "escalation_level": "L0",
    "latency_ms": 2,
    "conflict_detections": []
}
```

---

## 7. Configuration

### 7.1 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GGFL_ENABLED` | `false` | Enable governance fusion layer |
| `GGFL_MODE` | `shadow` | `shadow` or `active` |
| `GGFL_DEFAULT_ALLOW_BIAS` | `10.0` | Bias toward ALLOW in voting |
| `GGFL_STALENESS_TOLERANCE_SEC` | `60` | Max signal staleness |
| `GGFL_LOG_PATH` | `logs/fusion/` | Fusion log directory |

### 7.2 Tunable Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `default_allow_bias` | 10.0 | [0, 100] | Voting bias toward ALLOW |
| `staleness_window_cycles` | 1 | [1, 10] | Max cycles before signal stale |
| `conflict_threshold` | 0.3 | [0, 1] | Threshold for conflict detection |
| `escalation_cooldown_cycles` | 5 | [1, 20] | Cycles before de-escalation |

---

## 8. Phase X Constraints

During Phase X (SHADOW MODE), the following constraints apply:

1. **No Enforcement**: Fusion decisions are logged but never enforced
2. **No Modification**: Real governance decisions are unaffected
3. **Observational Only**: All outputs are for analysis
4. **Reversible**: GGFL can be disabled via `GGFL_ENABLED=false`
5. **Divergence Tracking**: Differences between fusion and real governance are logged

---

## 9. Future Phases

### Phase XI: Active Mode (Requires Authorization)

- Fusion decisions may influence real governance
- Requires explicit activation and monitoring period
- Rollback capability must be verified

### Phase XII: Full Integration

- Fusion layer becomes primary governance decision source
- All subsystems report through GGFL
- Complete audit trail requirement

---

## 10. NCI → GGFL Integration

The Narrative Consistency Index (NCI) provides documentation health metrics that feed into the GGFL through the Narrative Signal (SIG-NAR).

### 10.1 NCI Signal Mapping

The NCI governance signal is transformed into SIG-NAR fields as follows:

| NCI Field | SIG-NAR Field | Mapping |
|-----------|---------------|---------|
| `global_nci` | `narrative_coherence` | Direct mapping [0,1] |
| `slo_status` | `curriculum_health` | `OK→HEALTHY`, `WARN→DEGRADED`, `BREACH→CRITICAL` |
| `telemetry_consistency.drift_detected` | `drift_detected` | Direct boolean |
| `dominant_area` | (advisory) | Logged but not mapped |

### 10.2 NCI Governance Semantics in GGFL

| NCI Condition | GGFL Recommendation | Priority |
|---------------|---------------------|----------|
| `global_nci < 0.60` | WARNING | 3 |
| `slo_status == BREACH` | WARNING | 4 |
| `telemetry_consistency.drift_detected == true` | WARNING | 2 |
| `slice_consistency.aligned == false` | WARNING | 2 |
| Multiple WARN conditions (≥3) | SOFT BLOCK | 5 |

### 10.3 NCI Integration Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    NCI → GGFL Data Flow                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Documentation Files                                           │
│         │                                                       │
│         ▼                                                       │
│   ┌─────────────────┐                                          │
│   │ NCI Indexer     │                                          │
│   └────────┬────────┘                                          │
│            │                                                    │
│   ┌────────┴────────┐                                          │
│   │ NCI Director    │                                          │
│   │ Panel Builder   │                                          │
│   └────────┬────────┘                                          │
│            │                                                    │
│   ┌────────┴────────┐                                          │
│   │ NCI Governance  │                                          │
│   │ Signal Builder  │                                          │
│   └────────┬────────┘                                          │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────────────────────────────────────┐          │
│   │           GGFL: SIG-NAR Adapter                 │          │
│   │                                                 │          │
│   │  • Map global_nci → narrative_coherence        │          │
│   │  • Map slo_status → curriculum_health          │          │
│   │  • Map telemetry_drift → drift_detected        │          │
│   │  • Extract recommendations                      │          │
│   │                                                 │          │
│   └────────────────────┬────────────────────────────┘          │
│                        │                                        │
│                        ▼                                        │
│   ┌─────────────────────────────────────────────────┐          │
│   │           GGFL Fusion Layer                     │          │
│   │                                                 │          │
│   │  Signals: SIG-TOP, SIG-RPL, SIG-MET, ...       │          │
│   │           SIG-NAR (includes NCI)                │          │
│   │                                                 │          │
│   │  → Fusion Decision                              │          │
│   │  → Escalation Level                             │          │
│   │                                                 │          │
│   └─────────────────────────────────────────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 10.4 NCI Consistency Laws in GGFL Context

The NCI enforces two sets of consistency laws that contribute to GGFL narrative health:

**Telemetry Consistency Laws (TCL)**:
- TCL-001: Event name alignment
- TCL-002: Field name consistency (H, rho, tau, beta, in_omega)
- TCL-003: Schema version reference
- TCL-004: Telemetry drift synchronization

**Slice Identity Consistency Laws (SIC)**:
- SIC-001: Slice name canonicalization
- SIC-002: Slice parameter accuracy
- SIC-003: Slice phase mapping
- SIC-004: Slice capability claims

Violations of these laws contribute to NCI score degradation, which surfaces in GGFL as reduced `narrative_coherence` and potential WARNING recommendations.

### 10.5 NCI in Evidence Pack

The NCI signal is attached to the Phase X evidence pack at:
```
evidence["governance"]["nci"]
```

This enables whitepaper evidence bundles to include narrative health attestation alongside other governance signals.

### 10.6 NCI Update Frequency

| Context | Update Frequency |
|---------|------------------|
| CI pipeline | Per-commit |
| Nightly derivation | Per-run |
| Evidence pack generation | Per-pack |
| GGFL cycle integration | Per-epoch |

### 10.7 Coherence Signal Integration

The **Coherence Signal** provides a structural consistency assessment between decoy confusability geometry and topology health. This signal feeds into GGFL as a mild narrative/structure consistency indicator within the SIG-NAR context.

**Coherence Signal Mapping:**

The coherence governance tile is conceptually mapped into SIG-NAR as a structural consistency contributor:

| Coherence Field | SIG-NAR Contribution | Mapping |
|----------------|---------------------|---------|
| `coherence_band` | `narrative_coherence` | Indirect influence: COHERENT→higher coherence, MISALIGNED→lower coherence |
| `slices_at_risk` | (advisory) | Logged as structural consistency notes |
| `global_coherence_index` | (advisory) | Logged but not directly mapped |

**Coherence Governance Semantics in GGFL:**

| Coherence Condition | GGFL Recommendation | Priority |
|---------------------|---------------------|----------|
| `coherence_band == MISALIGNED` | WARNING | 2 |
| `coherence_band == PARTIAL` | (no recommendation) | N/A |
| `coherence_band == COHERENT` | (no recommendation) | N/A |
| `len(slices_at_risk) >= 3` | WARNING | 2 |

**Advisory Nature:**

The coherence signal is **advisory only** in Phase X. It does not produce hard blocks or mandatory recommendations. Instead, it serves as a cross-check signal that:

- Indicates structural alignment between decoy design and topology health
- Provides context for narrative consistency assessment
- Flags potential misalignment patterns for human review

**Coherence in Evidence Pack:**

The coherence signal is attached to the Phase X evidence pack at:
```
evidence["governance"]["coherence"]["first_light_summary"]
```

This enables whitepaper evidence bundles to include coherence assessment alongside other governance signals. The `first_light_summary` contains:
- `coherence_band`: "COHERENT" | "PARTIAL" | "MISALIGNED"
- `global_index`: float [0, 1]
- `slices_at_risk`: List[str]

**Integration Flow:**

```
Coherence Map (from decoy-topology analysis)
        │
        ▼
Coherence Governance Tile
        │
        ▼
GGFL: SIG-NAR Context (advisory contribution)
        │
        ▼
Evidence Pack: coherence.first_light_summary
```

### 10.8 Coherence CAL-EXP Snapshots (P5)

In P5 calibration experiments (CAL-EXP-1/2/3), coherence snapshots are generated per experiment and contribute to SIG-NAR analysis. Each calibration experiment produces a coherence snapshot (`calibration/coherence_<cal_id>.json`) that captures the structural consistency state at that calibration point.

**CAL-EXP Snapshot Structure:**

Each snapshot contains:
- `schema_version`: "1.0.0"
- `cal_id`: Calibration experiment identifier
- `coherence_band`: "COHERENT" | "PARTIAL" | "MISALIGNED"
- `global_index`: float [0, 1]
- `num_slices_at_risk`: int

**P5 SIG-NAR Contribution:**

Coherence CAL-EXP snapshots feed into P5 analysis by:
1. **Temporal Pattern Detection**: Comparing coherence bands across CAL-EXP-1/2/3 reveals structural consistency trends during calibration
2. **Cross-Experiment Validation**: If coherence degrades across experiments (COHERENT → PARTIAL → MISALIGNED), this contributes to `narrative_coherence` assessment
3. **Fusion Consistency Cross-Check**: The `summarize_coherence_vs_fusion()` function analyzes consistency between coherence signals and GGFL fusion results, identifying TENSION or CONFLICT patterns

**P5 Integration Flow:**

```
CAL-EXP-1/2/3 First Light Summaries
        │
        ▼
Coherence Snapshots (per experiment)
        │
        ▼
Coherence vs Fusion Cross-Check
        │
        ▼
GGFL: SIG-NAR (temporal coherence contribution)
        │
        ▼
Evidence Pack: coherence_fusion_crosscheck
```

**Advisory Nature:**

Coherence CAL-EXP snapshots are **advisory only** in P5. They provide structural consistency context for calibration analysis but do not gate calibration decisions. The cross-check analysis (`coherence_fusion_crosscheck`) flags consistency patterns (CONSISTENT, TENSION, CONFLICT) for human review but does not enforce hard blocks.

---

## 11. P5 Telemetry & Divergence Integration

This section specifies how P5 real telemetry validation and divergence patterns will integrate with GGFL signals. This is **forward-looking specification only** — no implementation is authorized until P5 activation.

### 11.1 P5 Divergence Pattern Taxonomy

P5 defines six divergence patterns when twin model predictions diverge from real telemetry (see Real_Telemetry_Topology_Spec.md Section 3.1):

| Pattern | Signature | Severity |
|---------|-----------|----------|
| `DRIFT` | Systematic bias, `mean(Δp) > 0.05, std(Δp) < 0.02` | MEDIUM |
| `NOISE_AMPLIFICATION` | Twin over-sensitive, `std(Δp) > 2 × std(p_real)` | LOW |
| `PHASE_LAG` | Temporal misalignment, `argmax(xcorr) ≠ 0` | LOW |
| `ATTRACTOR_MISS` | Frequent `ω_twin ≠ ω_real` | HIGH |
| `TRANSIENT_MISS` | High Δp during excursions only | MEDIUM |
| `STRUCTURAL_BREAK` | Sudden regime change, Δp stays high | CRITICAL |

### 11.2 Signal Slot Mapping

P5 divergence patterns will be mapped into existing GGFL signals as follows:

#### 11.2.1 SIG-TEL (Telemetry Signal) Extensions

| New Field | Type | Source | Description |
|-----------|------|--------|-------------|
| `telemetry_validation_status` | enum | P5 validator | `VALIDATED_REAL`, `SUSPECTED_MOCK`, `VALIDATION_PENDING` |
| `validation_confidence` | float [0,1] | P5 validator | Confidence in telemetry authenticity |
| `divergence_pattern` | enum | P5 classifier | Current dominant divergence pattern |
| `divergence_pattern_streak` | int | P5 classifier | Consecutive cycles with same pattern |
| `recalibration_triggered` | bool | P5 calibrator | Whether recalibration is in progress |

**Extended SIG-TEL Schema (P5)**:
```json
{
  "signal_id": "SIG-TEL",
  "lean_healthy": true,
  "db_healthy": true,
  "redis_healthy": true,
  "worker_count": 4,
  "error_rate": 0.01,
  "last_error": null,
  "uptime_seconds": 86400,
  "p5_telemetry": {
    "telemetry_validation_status": "VALIDATED_REAL",
    "validation_confidence": 0.92,
    "divergence_pattern": "DRIFT",
    "divergence_pattern_streak": 3,
    "recalibration_triggered": false
  }
}
```

#### 11.2.2 SIG-TOP (Topology Signal) Extensions

| New Field | Type | Source | Description |
|-----------|------|--------|-------------|
| `attractor_miss_rate` | float [0,1] | P5 divergence analyzer | Rate of `ω_twin ≠ ω_real` |
| `twin_omega_alignment` | bool | P5 twin model | Whether twin correctly predicts safe region |
| `transient_tracking_quality` | float [0,1] | P5 analyzer | How well twin tracks transients |

**Extended SIG-TOP Schema (P5)**:
```json
{
  "signal_id": "SIG-TOP",
  "H": 0.85,
  "rho": 0.87,
  "within_omega": true,
  "p5_twin": {
    "attractor_miss_rate": 0.05,
    "twin_omega_alignment": true,
    "transient_tracking_quality": 0.88
  }
}
```

#### 11.2.3 SIG-RPL (Replay Signal) Extensions

| New Field | Type | Source | Description |
|-----------|------|--------|-------------|
| `twin_prediction_divergence` | float [0,1] | P5 twin model | Mean `|p_twin - p_real|` |
| `divergence_bias` | float [-1,1] | P5 analyzer | Systematic bias direction |
| `divergence_variance` | float | P5 analyzer | Variance of divergence |

**Extended SIG-RPL Schema (P5)**:
```json
{
  "signal_id": "SIG-RPL",
  "replay_verified": true,
  "replay_divergence": 0.02,
  "replay_hash_match": true,
  "p5_divergence": {
    "twin_prediction_divergence": 0.04,
    "divergence_bias": 0.01,
    "divergence_variance": 0.008
  }
}
```

### 11.3 P5 Divergence Governance Semantics

#### 11.3.1 Pattern-Specific Recommendations

| Divergence Pattern | GGFL Recommendation | Priority | Reason |
|--------------------|---------------------|----------|--------|
| `DRIFT` | WARNING | 4 | Twin calibration may need adjustment |
| `NOISE_AMPLIFICATION` | WARNING | 3 | Twin over-fitting to noise |
| `PHASE_LAG` | WARNING | 3 | Prediction timing misalignment |
| `ATTRACTOR_MISS` | BLOCK | 7 | Twin fundamentally misaligned |
| `TRANSIENT_MISS` | WARNING | 5 | Transient fidelity concern |
| `STRUCTURAL_BREAK` | BLOCK | 8 | Regime change detected |

#### 11.3.2 Streak-Based Escalation

Persistent patterns trigger escalation:

| Condition | Recommendation | Priority |
|-----------|----------------|----------|
| `DRIFT` streak ≥ 5 cycles | BLOCK | 6 |
| `NOISE_AMPLIFICATION` streak ≥ 10 cycles | WARNING | 4 |
| `PHASE_LAG` streak ≥ 10 cycles | WARNING | 4 |
| `ATTRACTOR_MISS` streak ≥ 3 cycles | HARD_BLOCK | 9 |
| `TRANSIENT_MISS` streak ≥ 5 cycles | BLOCK | 6 |
| `STRUCTURAL_BREAK` streak ≥ 2 cycles | HARD_BLOCK | 10 |

### 11.4 Cross-Signal Consistency Rules (P5 Extensions)

**Rule CSC-P5-001**: If `SIG-TEL.p5_telemetry.divergence_pattern = STRUCTURAL_BREAK` AND `SIG-STR.min_cut_capacity < 0.2` (TENSION), escalate to L3 CRITICAL.

```
IF SIG-TEL.p5_telemetry.divergence_pattern == "STRUCTURAL_BREAK"
   AND SIG-TEL.p5_telemetry.divergence_pattern_streak >= 2
   AND SIG-STR.min_cut_capacity < 0.2
THEN
   escalate_to(L3_CRITICAL)
   reason = "Structural break with DAG tension: regime change under structural stress"
```

**Rule CSC-P5-002**: If `SIG-TEL.p5_telemetry.telemetry_validation_status = SUSPECTED_MOCK` AND `SIG-RPL.p5_divergence.twin_prediction_divergence < 0.01`, escalate as CONFLICT_DETECTED (suspiciously perfect twin alignment).

**Rule CSC-P5-003**: If `SIG-TOP.p5_twin.attractor_miss_rate > 0.2` AND `SIG-TOP.within_omega = true`, escalate as CONFLICT_DETECTED (twin fails to track safe region despite real system being safe).

**Rule CSC-P5-004**: If `SIG-TEL.p5_telemetry.recalibration_triggered = true` AND `SIG-BUD.compute_budget_remaining < 0.2`, escalate as CONFLICT_DETECTED (recalibration needed but insufficient budget).

### 11.5 P5 Escalation Rule (Primary)

The primary P5-specific escalation rule:

```
RULE P5-ESC-001: Structural Break Under Tension

TRIGGER:
  SIG-TEL.p5_telemetry.divergence_pattern == "STRUCTURAL_BREAK"
  AND SIG-TEL.p5_telemetry.divergence_pattern_streak >= 2
  AND SIG-STR.min_cut_capacity < 0.2

ACTION:
  escalation_level = L3_CRITICAL
  recommendation = HARD_BLOCK
  priority = 10

RATIONALE:
  A structural break indicates the real system has undergone a regime change
  that the twin model cannot predict. When this occurs under DAG structural
  tension (low min-cut capacity), the combination suggests:

  1. The proof graph is approaching a critical transition point
  2. The twin model has lost predictive validity
  3. Governance decisions based on twin predictions are unreliable

  This warrants immediate L3 CRITICAL escalation to halt derivation
  until the regime change is understood and twin recalibration completes.
```

### 11.6 P5 Signal Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    P5 → GGFL Signal Integration                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Real Telemetry Stream                                                  │
│         │                                                                │
│         ▼                                                                │
│   ┌─────────────────────┐                                               │
│   │ P5 Validator        │                                               │
│   │ (RTTS validation)   │                                               │
│   └────────┬────────────┘                                               │
│            │                                                             │
│            ├── telemetry_validation_status ─────► SIG-TEL.p5_telemetry  │
│            │                                                             │
│   ┌────────┴────────────┐                                               │
│   │ P5 Twin Model       │                                               │
│   │ (prediction)        │                                               │
│   └────────┬────────────┘                                               │
│            │                                                             │
│            ├── twin_prediction_divergence ──────► SIG-RPL.p5_divergence │
│            ├── twin_omega_alignment ────────────► SIG-TOP.p5_twin       │
│            │                                                             │
│   ┌────────┴────────────┐                                               │
│   │ P5 Divergence       │                                               │
│   │ Classifier          │                                               │
│   └────────┬────────────┘                                               │
│            │                                                             │
│            ├── divergence_pattern ──────────────► SIG-TEL.p5_telemetry  │
│            ├── divergence_pattern_streak ───────► SIG-TEL.p5_telemetry  │
│            ├── attractor_miss_rate ─────────────► SIG-TOP.p5_twin       │
│            │                                                             │
│            ▼                                                             │
│   ┌─────────────────────────────────────────────┐                       │
│   │           GGFL Fusion Layer                 │                       │
│   │                                             │                       │
│   │  • Extract P5 recommendations              │                       │
│   │  • Apply CSC-P5-* consistency rules        │                       │
│   │  • Evaluate P5-ESC-001 escalation          │                       │
│   │  • Merge with other signal recommendations │                       │
│   │                                             │                       │
│   └─────────────────────────────────────────────┘                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 11.7 P5 Phase Gating

This integration is gated by P5 activation:

| Gate | Condition | Status |
|------|-----------|--------|
| P5-GATE-001 | P3/P4 shadow validation complete | Pending |
| P5-GATE-002 | Real telemetry stream available | Pending |
| P5-GATE-003 | Twin calibration converged | Pending |
| P5-GATE-004 | GGFL P5 extensions implemented | Spec only |

Until all gates pass, P5 telemetry fields remain `null` and P5-specific rules are skipped.

---

## 12. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-10 | Initial Phase X specification |
| 1.1.0 | 2025-12-11 | Added NCI → GGFL integration (Section 10) |
| 1.2.0 | 2025-12-11 | Added P5 Telemetry & Divergence Integration (Section 11) |
| 1.3.0 | 2025-12-11 | Added Coherence Signal Integration (Section 10.7) |
