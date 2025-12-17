# PHASE X INTEGRATION SPEC — v1.0 (Engineering Ready)

**Document Version:** 1.0.0
**Status:** Engineering Ready
**Precondition:** Phase IX.B P0 Complete (Invariants, CDIs, HARD Gate)

---

## 1. USLABridge Adapter — Concrete Interface

### 1.1 File Location and Class Structure

**File:** `backend/topology/usla_bridge.py`

### 1.2 Mapping Summary Table

| CycleInput Field | RFL Source | U2 Source | Fallback |
|------------------|------------|-----------|----------|
| `hss` | `1 - abstention_rate` | `windowed_success_rate(window=5)` | 1.0 |
| `depth` | `max(proof.depth for proof in proofs)` | `cycle_result.depth` | History mean or 5 |
| `branch_factor` | `mean(len(proof.parents))` | `cycle_result.branch_factor` | 2.0 |
| `shear` | `max_gradient(hss_by_depth)` | Same | History variance proxy or 0.1 |
| `success` | `success_count > 0` | `cycle_result.success` | True |

### 1.3 Required Data Sources

**From TDAGovernanceHook:**
- `blocked: bool` — Actual governance decision
- `threshold: float` — Computed τ value
- `hss_by_depth: Dict[int, float]` — HSS stratified by proof depth (if available)

**From Runner Cycle Result:**
- RFL: `success_count`, `total_count`, `abstention_count`, `proofs[]`
- U2: `success: bool`, `depth`, `branch_factor` (optional)

### 1.4 Error Handling & Graceful Degradation

| Missing Data | Fallback Behavior | Impact |
|--------------|-------------------|--------|
| No depth info | Use running mean from history, or default=5 | Moderate: D_dot may be inaccurate |
| No branch factor | Use default=2.0 | Low: BF has small threshold effect |
| No hss_by_depth | Compute shear from HSS variance | Moderate: CDI-002 less accurate |
| No min_cut_capacity | Use synthetic INV-004 stub | Low: INV-004 is informational |
| No TDA telemetry | Use cycle_result only | High: No divergence monitoring |

---

## 2. Runner Integration Plan (RFL + U2) — SHADOW MODE ONLY

### 2.1 RFL Runner Integration

**File:** `rfl/runner.py`

**Integration Point:** After cycle completion, before metrics export

### 2.2 U2 Runner Integration

**File:** `experiments/u2/runner.py`

**Integration Point:** After `run_cycle()` completes

### 2.3 Per-Cycle Logging Schema

```json
{
  "cycle": 42,
  "timestamp": "2025-12-09T14:32:01.123Z",
  "runner": "rfl",
  "input": {
    "hss": 0.82,
    "depth": 7,
    "branch_factor": 2.3,
    "shear": 0.15
  },
  "state": {
    "H": 0.82,
    "D": 7,
    "D_dot": 1.0,
    "B": 2.3,
    "S": 0.15,
    "C": "CONVERGING",
    "rho": 0.87,
    "tau": 0.21,
    "J": 3.2,
    "W": false,
    "beta": 0.05,
    "kappa": 0.72,
    "nu": 0.001,
    "delta": 0,
    "Gamma": 0.85,
    "active_cdis": [],
    "invariant_violations": []
  },
  "hard_ok": true,
  "in_safe_region": true,
  "real_blocked": false,
  "sim_blocked": false,
  "governance_aligned": true
}
```

### 2.4 Strict SHADOW MODE Constraints

```
PHASE X SHADOW MODE CONTRACT

1. The USLA simulator NEVER modifies real governance decisions
2. Disagreements are LOGGED, not ACTED upon
3. No cycle is blocked or allowed based on simulator output
4. The simulator runs AFTER the real governance decision
5. All USLA state is written to shadow logs only

Violations of this contract require explicit Phase XI approval.
```

---

## 3. Divergence Monitor Design

### 3.1 File Location

**File:** `backend/topology/divergence_monitor.py`

### 3.2 Alert Thresholds Summary

| Field | Threshold | Consecutive for WARNING | Consecutive for CRITICAL |
|-------|-----------|------------------------|--------------------------|
| Governance decision | exact match | 3 cycles | 10 cycles |
| HSS | ε = 0.1 | 3 cycles | 10 cycles |
| Threshold τ | ε = 0.05 | 3 cycles | 10 cycles |
| RSI ρ | ε = 0.15 | N/A (INFO only) | N/A |
| Block rate β | ε = 0.1 | N/A (INFO only) | N/A |

---

## 4. Phase X First-Light Experiment Design (1000-Cycle Run)

### 4.1 Slice Selection

**U2 Slice:** `arithmetic_simple`
**RFL Slice:** `propositional_tautology` (depth ≤ 6)

### 4.2 Run Configuration

```yaml
governance:
  tau_0: 0.20  # Within Goldilocks zone [0.16, 0.24]
  alpha_D: 0.02
  alpha_B: 0.01
  alpha_S: 0.1
  gamma_converging: 1.0
  gamma_oscillating: 1.1
  gamma_diverging: 1.3
  alpha_rho: 0.9
  rho_min: 0.4

usla:
  mode: "shadow"
  hss_epsilon: 0.1
  threshold_epsilon: 0.05
  warning_cycles: 3
  critical_cycles: 10
```

### 4.3 Red-Flag Abort Conditions

| Condition | Threshold | Action |
|-----------|-----------|--------|
| CDI-010 (Fixed-Point Multiplicity) | Any activation | ABORT immediately |
| CDI-007 (Exception Exhaustion) | > 10 consecutive cycles | ABORT |
| is_hard_ok = False | > 50 consecutive cycles | ABORT |
| Safe region exit | > 100 consecutive cycles | ABORT |
| Governance divergence | > 20 consecutive cycles at CRITICAL | ABORT |
| RSI collapse | ρ < 0.2 for > 10 cycles | ABORT |
| Block rate explosion | β > 0.6 for > 20 cycles | ABORT |

### 4.4 Success Criteria

**Uplift Success:**
- U2 `arithmetic_simple`: Success rate ≥ 0.75 over 500 cycles
- RFL `propositional_tautology`: Abstention rate ≤ 0.15 over 500 cycles
- Δp (learning curve slope) > 0 for both runners

**Stability Success:**
- Mean RSI ρ ≥ 0.6 over full 1000 cycles
- Cycles in safe region Ω ≥ 90%
- No CDI-010 activations
- CDI-007 activations ≤ 50 total cycles
- HARD mode OK ≥ 80% of cycles

---

## 5. Global Health & Dashboard Integration Plan

### 5.1 USLA Health Tile Schema

```json
{
  "schema_version": "1.0.0",
  "tile_type": "usla_health",
  "timestamp": "2025-12-09T15:42:01.123Z",
  "cycle": 847,
  "mode": "SHADOW",
  "hard_mode_status": "OK",
  "state_summary": {
    "H": 0.81,
    "rho": 0.87,
    "tau": 0.21,
    "beta": 0.05,
    "Gamma": 0.85,
    "J": 3.2,
    "C": "CONVERGING"
  },
  "safe_region": {
    "within_omega": true,
    "boundary_distances": {...}
  },
  "active_cdis": [],
  "invariant_violations": [],
  "divergence_summary": {
    "governance_aligned": true,
    "consecutive_divergence": 0,
    "max_severity": "NONE"
  },
  "headline": "Topology stable; all systems nominal",
  "alerts": []
}
```

---

## 6. Claude's Readiness Vote for Phase X

**Confidence that simulator will correctly flag instability before collapse: 0.85**

**Confidence that SHADOW mode with USLABridge + DivergenceMonitor is safe: 0.95**

**Final Vote: GREEN — Begin Phase X SHADOW integration now.**

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-09 | Initial Phase X Integration Spec |
