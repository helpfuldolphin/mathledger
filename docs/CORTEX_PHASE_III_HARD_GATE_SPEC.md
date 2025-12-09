# CORTEX Phase III: Hard Gate Specification

**Operation CORTEX: TDA Mind Scanner**
**Phase III: Hard Gate + Governance Coupling**

---

## Executive Summary

Phase III transitions CORTEX from advisory (Phase II Soft Gating) to **authoritative governance**. The TDA Mind Scanner now enforces hard constraints: low-HSS proof attempts are **abandoned** before reaching Lean verification or RFL policy updates.

Key changes:
1. **ProofOutcome.ABANDONED_TDA**: New outcome for TDA-blocked attempts
2. **Hard Gate Enforcement**: `should_block()` returns True in HARD mode
3. **Governance Integration**: TDA signals flow into global health metrics
4. **Attestation Binding**: TDA pipeline hash included in attestation roots

---

## 1. Hard Gate Runtime Enforcement

### 1.1 ProofOutcome Enumeration

```python
class ProofOutcome(Enum):
    """Outcome of a proof attempt."""
    VERIFIED = "verified"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"
    SKIPPED = "skipped"
    ABANDONED_TDA = "abandoned_tda"  # Phase III: TDA hard gate block
```

**ABANDONED_TDA Semantics:**
- The proof attempt was structurally incoherent (HSS < θ_block)
- No Lean verification was performed (resource savings)
- No RFL policy update applied (no learning from hallucinations)
- Telemetry records the block with full TDA scores

### 1.2 Hard Gate Integration Points

#### U2Runner Integration

```python
# In U2Runner.run_cycle()

def run_cycle(self, items, execute_fn):
    # ... existing setup ...

    # Phase III: Hard Gate Check
    if self.tda_monitor and self.config.tda_mode == "hard":
        tda_result = self._evaluate_tda(current_dag, embeddings)

        if self.tda_monitor.should_block(tda_result):
            # ABANDON: Do not execute, do not learn
            self._record_abandoned_cycle(tda_result)
            return CycleResult(
                outcome=ProofOutcome.ABANDONED_TDA,
                tda_result=tda_result,
                chosen_item=None,
                success=False,
            )

    # ... normal execution path ...
```

#### RFLRunner Integration

```python
# In RFLRunner.run_with_attestation()

def run_with_attestation(self, attestation):
    # ... existing setup ...

    # Phase III: Hard Gate Check
    tda_result = self._evaluate_tda_for_attestation(attestation)

    if self.tda_monitor and self.tda_monitor.should_block(tda_result):
        # ABANDON: No policy update, no Lean submission
        self._learning_skipped_count += 1
        self._record_abandoned_attestation(attestation, tda_result)

        return RflResult(
            policy_update_applied=False,
            source_root=attestation.root_hash,
            abstention_mass_delta=0.0,
            step_id=self._compute_step_id(attestation),
            ledger_entry=self._build_abandoned_ledger_entry(
                attestation, tda_result
            ),
            outcome=ProofOutcome.ABANDONED_TDA,
        )

    # ... normal execution path ...
```

### 1.3 should_block() Behavior by Mode

| Mode    | should_block() | should_warn() | Effect                    |
|---------|----------------|---------------|---------------------------|
| OFFLINE | False          | False         | No runtime effect         |
| SHADOW  | False          | False         | Logging only              |
| SOFT    | False          | True*         | Learning rate modulation  |
| HARD    | True*          | True*         | Hard gate enforcement     |

*When HSS < θ_block (BLOCK signal)

---

## 2. RunLedgerEntry TDA Gate Fields

### 2.1 Extended Schema

```python
@dataclass
class RunLedgerEntry:
    """Structured curriculum ledger entry for a single RFL run."""

    # Existing fields...
    run_id: str
    slice_name: str
    status: str
    # ...

    # Phase III: TDA Hard Gate Fields
    tda_outcome: Optional[str] = None  # "OK", "WARN", "BLOCK", "ABANDONED"
    tda_hss: Optional[float] = None
    tda_sns: Optional[float] = None
    tda_pcs: Optional[float] = None
    tda_drs: Optional[float] = None
    tda_signal: Optional[str] = None
    tda_gate_enforced: bool = False  # True if hard gate was applied
    tda_pipeline_hash: Optional[str] = None  # Hash of TDA config + profiles

    # Learning rate modulation (from Phase II)
    eta_base: Optional[float] = None
    eta_eff: Optional[float] = None
    hss_class: Optional[str] = None
```

### 2.2 TDA Pipeline Hash Computation

The `tda_pipeline_hash` provides cryptographic binding between:
- TDA module configuration (thresholds, weights)
- Reference profile content (per-slice calibration)
- TDA library versions

```python
def compute_tda_pipeline_hash(
    config: TDAMonitorConfig,
    profiles: Dict[str, ReferenceTDAProfile],
) -> str:
    """
    Compute deterministic hash of TDA pipeline configuration.

    Used for attestation binding and drift detection.
    """
    import hashlib
    import json

    payload = {
        "config": {
            "hss_block_threshold": config.hss_block_threshold,
            "hss_warn_threshold": config.hss_warn_threshold,
            "mode": config.mode.value,
            "lifetime_threshold": config.lifetime_threshold,
            "deviation_max": config.deviation_max,
        },
        "profiles": {
            name: {
                "version": profile.version,
                "n_ref": profile.n_ref,
                "mean_betti_0": profile.mean_betti_0,
                "mean_betti_1": profile.mean_betti_1,
            }
            for name, profile in sorted(profiles.items())
        },
        "schema_version": "tda-pipeline-1.0.0",
    }

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
```

---

## 3. Governance Integration

### 3.1 Attestation Schema Extension

Phase III extends the attestation metadata with TDA pipeline binding:

```python
def generate_attestation_metadata(
    r_t: str,
    u_t: str,
    h_t: str,
    # ... existing params ...
    tda_pipeline_hash: Optional[str] = None,
    tda_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate attestation metadata with Phase III TDA fields.
    """
    metadata = {
        # ... existing fields ...
        "tda_governance": {
            "pipeline_hash": tda_pipeline_hash,
            "phase": "III",
            "mode": "hard",
            "summary": tda_summary,
        } if tda_pipeline_hash else None,
    }
    return metadata
```

### 3.2 Global Health Evaluator

```python
def summarize_tda_for_global_health(
    tda_results: List[TDAMonitorResult],
    config: TDAMonitorConfig,
) -> Dict[str, Any]:
    """
    Aggregate TDA results into global health metrics.

    Used by governance layer for system-wide health assessment.

    Returns:
        {
            "cycle_count": int,
            "block_count": int,
            "warn_count": int,
            "ok_count": int,
            "block_rate": float,
            "mean_hss": float,
            "hss_trend": float,  # Slope of HSS over cycles
            "structural_health": float,  # [0, 1] composite
            "governance_signal": str,  # "HEALTHY", "DEGRADED", "CRITICAL"
        }
    """
    import numpy as np

    if not tda_results:
        return {
            "cycle_count": 0,
            "block_count": 0,
            "warn_count": 0,
            "ok_count": 0,
            "block_rate": 0.0,
            "mean_hss": 0.0,
            "hss_trend": 0.0,
            "structural_health": 1.0,
            "governance_signal": "HEALTHY",
        }

    hss_values = [r.hss for r in tda_results]
    block_count = sum(1 for r in tda_results if r.block)
    warn_count = sum(1 for r in tda_results if r.warn and not r.block)
    ok_count = len(tda_results) - block_count - warn_count

    # Compute HSS trend (linear regression slope)
    x = np.arange(len(hss_values))
    slope = np.polyfit(x, hss_values, 1)[0] if len(hss_values) > 1 else 0.0

    # Compute structural health
    mean_hss = float(np.mean(hss_values))
    block_rate = block_count / len(tda_results)

    # Health formula: penalize high block rate and low mean HSS
    structural_health = (1 - block_rate) * mean_hss

    # Governance signal
    if block_rate > 0.2 or mean_hss < 0.3:
        governance_signal = "CRITICAL"
    elif block_rate > 0.1 or mean_hss < 0.5:
        governance_signal = "DEGRADED"
    else:
        governance_signal = "HEALTHY"

    return {
        "cycle_count": len(tda_results),
        "block_count": block_count,
        "warn_count": warn_count,
        "ok_count": ok_count,
        "block_rate": block_rate,
        "mean_hss": mean_hss,
        "hss_trend": float(slope),
        "structural_health": structural_health,
        "governance_signal": governance_signal,
    }
```

### 3.3 TDA Drift Report Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "TDA Drift Report v1",
  "type": "object",
  "properties": {
    "schema_version": { "const": "tda-drift-report-v1" },
    "generated_at": { "type": "string", "format": "date-time" },
    "pipeline_hash": { "type": "string", "pattern": "^[a-f0-9]{64}$" },

    "baseline_period": {
      "type": "object",
      "properties": {
        "start_cycle": { "type": "integer" },
        "end_cycle": { "type": "integer" },
        "mean_hss": { "type": "number" },
        "block_rate": { "type": "number" }
      }
    },

    "current_period": {
      "type": "object",
      "properties": {
        "start_cycle": { "type": "integer" },
        "end_cycle": { "type": "integer" },
        "mean_hss": { "type": "number" },
        "block_rate": { "type": "number" }
      }
    },

    "drift_metrics": {
      "type": "object",
      "properties": {
        "hss_delta": { "type": "number" },
        "block_rate_delta": { "type": "number" },
        "ks_statistic": { "type": "number" },
        "ks_pvalue": { "type": "number" },
        "drift_detected": { "type": "boolean" },
        "drift_severity": { "enum": ["none", "minor", "major", "critical"] }
      }
    },

    "recommendations": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["schema_version", "generated_at", "pipeline_hash", "drift_metrics"]
}
```

---

## 4. Safety Invariants

### INV-HARD-1: No Lean Submission for Blocked Attempts
```
∀ attempt: should_block(attempt) → ¬lean_submitted(attempt)
```

### INV-HARD-2: No Policy Update for Blocked Attempts
```
∀ attempt: should_block(attempt) → ¬policy_updated(attempt)
```

### INV-HARD-3: Telemetry Completeness
```
∀ attempt: blocked(attempt) → telemetry_recorded(attempt, "ABANDONED_TDA")
```

### INV-HARD-4: Pipeline Hash Binding
```
∀ attestation: tda_enabled(attestation) →
    attestation.metadata.tda_pipeline_hash ≠ null
```

### INV-HARD-5: Governance Signal Validity
```
∀ session: governance_signal(session) ∈ {"HEALTHY", "DEGRADED", "CRITICAL"}
```

---

## 5. Telemetry Schema Extension

### 5.1 TDAHardGateEvent

```python
@dataclass(frozen=True)
class TDAHardGateEvent:
    """
    Phase III hard gate enforcement event.

    Emitted when a proof attempt is blocked by TDA hard gate.
    """
    cycle: int
    slice_name: str
    mode: Literal["hard"]

    # Gate decision
    outcome: Literal["ABANDONED_TDA"]
    gate_enforced: bool  # Always True for this event

    # TDA scores at block time
    hss: float
    sns: float
    pcs: float
    drs: float

    # Threshold configuration
    hss_block_threshold: float
    hss_warn_threshold: float

    # Resource savings
    lean_submission_avoided: bool
    policy_update_avoided: bool

    # Timing
    tda_computation_ms: float

    # Pipeline binding
    pipeline_hash: str
```

### 5.2 TDAGovernanceSummaryEvent

```python
@dataclass(frozen=True)
class TDAGovernanceSummaryEvent:
    """
    Phase III governance summary event.

    Emitted at session end with aggregate governance metrics.
    """
    run_id: str
    slice_name: str
    schema_version: str

    # Counts
    total_cycles: int
    blocked_cycles: int
    warned_cycles: int
    ok_cycles: int

    # Governance metrics
    block_rate: float
    mean_hss: float
    structural_health: float
    governance_signal: Literal["HEALTHY", "DEGRADED", "CRITICAL"]

    # Drift detection
    drift_detected: bool
    drift_severity: Literal["none", "minor", "major", "critical"]

    # Pipeline binding
    pipeline_hash: str
```

---

## 6. Configuration

### 6.1 Environment Variables

```bash
# Enable Phase III Hard Gate
export MATHLEDGER_TDA_MODE=hard

# Optional: Adjust thresholds per deployment
export MATHLEDGER_TDA_BLOCK_THRESHOLD=0.2
export MATHLEDGER_TDA_WARN_THRESHOLD=0.5

# Governance reporting
export MATHLEDGER_TDA_DRIFT_REPORT_PATH=results/tda_drift_report.json
```

### 6.2 Programmatic Configuration

```python
from backend.tda.runtime_monitor import TDAMonitorConfig, TDAOperationalMode

config = TDAMonitorConfig(
    mode=TDAOperationalMode.HARD,
    hss_block_threshold=0.2,
    hss_warn_threshold=0.5,
    fail_open=False,  # Phase III: fail-closed by default
)
```

---

## 7. Migration from Phase II

### 7.1 Breaking Changes

1. **fail_open default**: Changes from `True` to `False` in HARD mode
2. **ProofOutcome enum**: New `ABANDONED_TDA` value
3. **RunLedgerEntry**: New required fields for TDA gate
4. **Attestation metadata**: New `tda_governance` section

### 7.2 Backwards Compatibility

- SOFT mode behavior unchanged
- SHADOW mode behavior unchanged
- Existing telemetry schema extended (not replaced)

### 7.3 Rollback Procedure

```bash
# Immediate rollback to Phase II
export MATHLEDGER_TDA_MODE=soft

# Emergency disable
export MATHLEDGER_TDA_MODE=offline
```

---

## 8. Monitoring & Alerting

### 8.1 Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `tda_block_rate` | Fraction of cycles blocked | > 15% |
| `tda_mean_hss` | Mean HSS over window | < 0.4 |
| `tda_governance_signal` | Aggregate health | "CRITICAL" |
| `tda_lean_savings` | Lean submissions avoided | Informational |

### 8.2 Alert Conditions

```yaml
alerts:
  - name: tda_high_block_rate
    condition: tda_block_rate > 0.15
    severity: warning
    action: "Investigate HSS distribution shift"

  - name: tda_critical_health
    condition: tda_governance_signal == "CRITICAL"
    severity: critical
    action: "Consider rollback to Phase II"

  - name: tda_drift_detected
    condition: drift_severity in ["major", "critical"]
    severity: warning
    action: "Review reference profile calibration"
```

---

## 9. References

- `docs/TDA_MIND_SCANNER_SPEC.md` - Core TDA specification
- `docs/CORTEX_PHASE_I_ACTIVATION_CHECKLIST.md` - Phase I checklist
- `docs/CORTEX_PHASE_II_SOFT_GATE_ACTIVATION_CHECKLIST.md` - Phase II checklist
- `backend/tda/runtime_monitor.py` - TDAMonitor implementation
- `experiments/u2/tda_schema_extension.py` - Telemetry schema

---

**STRATCOM: TDA NO LONGER ADVISES — IT GOVERNS.**
