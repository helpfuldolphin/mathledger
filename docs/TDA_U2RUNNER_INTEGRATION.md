# TDA Mind Scanner - U2Runner Integration Notes

**Phase I: Shadow Mode Integration**

This document provides integration guidance for hooking `TDAMonitor` into the U2 Runner
for Phase I (Shadow Mode) deployment.

---

## 1. Integration Architecture

```
experiments/u2/runner.py
         |
         |  (1) After each cycle / proof attempt
         v
+------------------+
|   TDAMonitor     |  <-- Injected via constructor
+------------------+
         |
         |  (2) evaluate_proof_attempt()
         v
+------------------+
| TDAMonitorResult |  --> HSS, SNS, PCS, DRS, signal
+------------------+
         |
         |  (3) Log to H_t series telemetry
         v
   ht_series.append({
       "tda_hss": result.hss,
       "tda_signal": result.signal.value,
       ...
   })
```

---

## 2. Code Changes Required

### 2.1 Modify U2Runner Constructor

```python
# experiments/u2/runner.py

from typing import Optional
from backend.tda.runtime_monitor import TDAMonitor, TDAMonitorResult

class U2Runner:
    def __init__(
        self,
        ...,
        tda_monitor: Optional[TDAMonitor] = None,  # NEW
    ):
        ...
        self.tda_monitor = tda_monitor
```

### 2.2 Add TDA Evaluation Hook

Insert after each cycle's proof attempt (before RFL update):

```python
def _run_cycle(self, cycle: int, ...):
    # ... existing cycle logic ...

    # Extract local DAG around the target
    if self.tda_monitor is not None:
        from backend.tda.proof_complex import extract_local_neighborhood, dag_from_proof_dag

        # Get local DAG (depth-3 neighborhood)
        local_dag = self._extract_local_dag(target_hash)

        # Get state embeddings from recent states
        embeddings = self._get_state_embeddings()

        # Evaluate
        tda_result = self.tda_monitor.evaluate_proof_attempt(
            slice_name=self.slice_name,
            local_dag=local_dag,
            embeddings=embeddings,
        )

        # Log to telemetry
        self._log_tda_result(cycle, tda_result)

        # Phase I: Shadow mode - no gating, just observe
        # Future phases will gate here
```

### 2.3 Helper Methods

```python
def _extract_local_dag(self, target_hash: str) -> "nx.DiGraph":
    """Extract local proof neighborhood for TDA analysis."""
    from backend.tda.proof_complex import (
        extract_local_neighborhood,
        dag_from_proof_dag,
    )

    # Convert ProofDag to NetworkX
    nx_dag = dag_from_proof_dag(self.proof_dag)

    # Extract bounded neighborhood
    return extract_local_neighborhood(
        nx_dag,
        target_hash,
        max_depth=3,
        include_descendants=True,
    )

def _get_state_embeddings(self) -> Dict[str, np.ndarray]:
    """Get embeddings for recent reasoning states."""
    from backend.axiom_engine.features import extract_statement_features

    embeddings = {}
    # Use last N states from cycle history
    for i, state in enumerate(self._recent_states[-50:]):
        key = f"state_{i}"
        embeddings[key] = extract_statement_features(state.text)

    return embeddings

def _log_tda_result(self, cycle: int, result: TDAMonitorResult) -> None:
    """Log TDA result to H_t series telemetry."""
    self.ht_series.append({
        "cycle": cycle,
        "event_type": "tda_evaluation",
        "tda_hss": result.hss,
        "tda_sns": result.sns,
        "tda_pcs": result.pcs,
        "tda_drs": result.drs,
        "tda_signal": result.signal.value,
        "tda_block": result.block,
        "tda_warn": result.warn,
        "tda_betti_0": result.betti.get(0, 0),
        "tda_betti_1": result.betti.get(1, 0),
        "tda_computation_ms": result.computation_time_ms,
    })
```

---

## 3. Instantiation at Experiment Level

```python
# experiments/u2/runtime/experiment_runner.py or orchestrator

from backend.tda.runtime_monitor import (
    TDAMonitor,
    TDAMonitorConfig,
    TDAOperationalMode,
)
from backend.tda.reference_profile import load_reference_profiles
from pathlib import Path

def create_u2_runner_with_tda(
    slice_name: str,
    mode: str = "shadow",
    profiles_path: Optional[Path] = None,
) -> U2Runner:
    # Load reference profiles if available
    profiles = {}
    if profiles_path and profiles_path.exists():
        profiles = load_reference_profiles(profiles_path)

    # Configure TDA monitor
    tda_config = TDAMonitorConfig(
        hss_block_threshold=0.2,
        hss_warn_threshold=0.5,
        mode=TDAOperationalMode(mode),
        fail_open=True,  # Safe for Phase I
    )

    tda_monitor = TDAMonitor(tda_config, profiles)

    # Create runner with monitor
    return U2Runner(
        ...,
        tda_monitor=tda_monitor,
    )
```

---

## 4. Phase I Validation Checklist

Before advancing to Phase II (Soft Gating):

- [ ] TDA monitor successfully evaluates all proof attempts
- [ ] HSS scores are logged to H_t series
- [ ] No runtime errors or crashes from TDA computation
- [ ] Computation time < 100ms per evaluation (95th percentile)
- [ ] HSS distribution correlates with Lean verification outcomes
- [ ] HSS distribution correlates with RFL convergence metrics
- [ ] Reference profiles built for active slices

---

## 5. Telemetry Schema Extension

Add to `experiments/u2/schema.py`:

```python
@dataclass
class TDAEvaluationEvent:
    """TDA Mind Scanner evaluation result."""
    cycle: int
    hss: float
    sns: float
    pcs: float
    drs: float
    signal: str  # "BLOCK", "WARN", "OK"
    betti_0: int
    betti_1: int
    computation_ms: float
    slice_name: str
```

---

## 6. Phase II Transition

When advancing to Soft Gating:

1. Change mode from `SHADOW` to `SOFT`
2. Modify RFL update to check `should_warn()`:

```python
if self.tda_monitor is not None:
    result = self.tda_monitor.evaluate_proof_attempt(...)

    if self.tda_monitor.should_warn(result):
        # Downweight learning rate
        learning_rate *= 0.5  # or configurable factor

        # Log anomaly
        self._log_tda_anomaly(cycle, result)
```

---

## 7. Phase III Transition (Hard Gating)

When advancing to Hard Gating:

1. Change mode from `SOFT` to `HARD`
2. Add branch pruning:

```python
if self.tda_monitor is not None:
    result = self.tda_monitor.evaluate_proof_attempt(...)

    if self.tda_monitor.should_block(result):
        # Abandon this proof attempt
        self._log_tda_block(cycle, result)
        return ProofOutcome.ABANDONED_TDA

    if self.tda_monitor.should_warn(result):
        # Continue but downweight
        ...
```

---

## 8. Performance Considerations

- **Limit local DAG depth**: Use `max_depth=3` for neighborhood extraction
- **Limit embedding window**: Use last 50-200 states
- **Cache reference profiles**: Load once at startup
- **Async option**: Run TDA in separate thread if latency is critical

---

## 9. Monitoring & Observability

Key metrics to track:

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `tda_eval_latency_p95` | 95th percentile computation time | > 100ms |
| `tda_error_rate` | Fraction of evaluations that error | > 1% |
| `tda_block_rate` | Fraction of BLOCK signals (Hard mode) | Context-dependent |
| `tda_hss_mean` | Mean HSS across evaluations | < 0.3 (investigate) |

---

## 10. Rollback Procedure

If TDA integration causes issues:

1. Set `tda_monitor = None` in runner instantiation
2. Or set mode to `OFFLINE` (no evaluation)
3. All TDA code paths have null-checks and fail gracefully

---

**Document Version:** 1.0
**Spec Reference:** TDA_MIND_SCANNER_SPEC.md v0.1
**Phase:** I (Shadow Mode)
