# Phase X P3 Specification: First-Light Shadow Experiment Architecture

> **This document is a DESIGN-ONLY specification. Implementation requires explicit future authorization.**

**Status**: Design Freeze (Stubs Only)
**Phase**: X P3 (SHADOW MODE ONLY)
**Version**: 1.0.0
**Date**: 2025-12-09

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [First-Light Shadow Experiment Runner API](#2-first-light-shadow-experiment-runner-api)
3. [Red-Flag Observation Layer](#3-red-flag-observation-layer)
4. [Δp Computation Contract](#4-Δp-computation-contract)
5. [JSONL Log Schemas](#5-jsonl-log-schemas)
6. [Example 50-Cycle Shadow Test Output](#6-example-50-cycle-shadow-test-output)
7. [Safety Boundaries](#7-safety-boundaries)
8. [CI Requirements](#8-ci-requirements)
9. [Implementation Plan](#9-implementation-plan)
10. [Authorization Gates](#10-authorization-gates)
11. [Summary](#11-summary)

---

## 1. Executive Summary

Phase X P3 defines the architectural scaffolding for the First-Light 1000-cycle shadow experiment. This specification covers:

1. **First-Light Shadow Runner API** — Input/output contracts for experiment execution
2. **Red-Flag Observation Layer** — Logging-only detection of anomalous conditions
3. **Δp Computation Contract** — Learning curve metrics in SHADOW mode
4. **JSONL Log Schemas** — Structured output formats
5. **Implementation Plan** — Files, functions, tests, CI

### SHADOW MODE CONTRACT

All components are observational only. No governance decisions are modified. No abort logic influences control flow.

| Invariant | Description |
|-----------|-------------|
| No governance modification | USLA outputs never influence real decisions |
| No abort enforcement | Red-flags are logged, never acted upon |
| Observational only | All outputs are for analysis/logging |
| Reversible | Can be disabled via environment variable |

---

## 2. First-Light Shadow Experiment Runner API

### 2.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FirstLightShadowRunner                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│  │  FirstLightConfig │    │  SliceExecutor   │    │  MetricsWindow   │      │
│  │                   │───▶│   (U2 / RFL)     │───▶│                  │      │
│  │  - slice_name     │    │                  │    │  - success_rates │      │
│  │  - runner_type    │    │  SHADOW MODE:    │    │  - abstention    │      │
│  │  - total_cycles   │    │  Observes only,  │    │  - rsi_history   │      │
│  │  - tau_0          │    │  never modifies  │    │  - block_rates   │      │
│  │  - log_dir        │    │  governance      │    │                  │      │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘      │
│           │                       │                       │                 │
│           ▼                       ▼                       ▼                 │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                     USLAIntegration (SHADOW)                      │      │
│  │                                                                   │      │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │      │
│  │  │ USLABridge  │  │ ShadowLogger│  │ Divergence  │              │      │
│  │  │             │  │             │  │  Monitor    │              │      │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│           │                       │                       │                 │
│           ▼                       ▼                       ▼                 │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                   RedFlagObservationLayer                         │      │
│  │                                                                   │      │
│  │  Observes (NEVER enforces):                                      │      │
│  │  - CDI-010 activations        - RSI collapse (ρ < 0.2)          │      │
│  │  - CDI-007 streaks            - Block rate explosion (β > 0.6)  │      │
│  │  - Safe region (Ω) exits      - Threshold drift                 │      │
│  │  - HARD mode failures         - Governance divergence           │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                      DeltaPComputer                               │      │
│  │                                                                   │      │
│  │  Computes (read-only):                                           │      │
│  │  - Δp_success: U2 success rate slope                            │      │
│  │  - Δp_abstention: RFL abstention reduction slope                │      │
│  │  - Ω_occupancy: Safe region occupancy curve                     │      │
│  │  - HARD_curve: HARD-OK percentage over time                     │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                       Output Artifacts                            │      │
│  │                                                                   │      │
│  │  - results/first_light/{run_id}/cycles.jsonl                    │      │
│  │  - results/first_light/{run_id}/red_flags.jsonl                 │      │
│  │  - results/first_light/{run_id}/metrics.jsonl                   │      │
│  │  - results/first_light/{run_id}/summary.json                    │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Input Contract

```python
@dataclass
class FirstLightConfig:
    """Configuration for First-Light shadow experiment."""

    # Slice selection
    slice_name: str                    # "arithmetic_simple" or "propositional_tautology"
    runner_type: RunnerType            # RunnerType.U2 or RunnerType.RFL

    # Run parameters
    total_cycles: int = 1000           # Target cycle count
    tau_0: float = 0.20                # Initial threshold (Goldilocks: [0.16, 0.24])

    # USLA parameters (optional override)
    usla_params: Optional[USLAParams] = None

    # Logging configuration
    log_dir: str = "results/first_light"
    run_id: Optional[str] = None       # Auto-generated if None
    log_every_n_cycles: int = 1        # Log frequency

    # Metrics windows
    success_window: int = 50           # Window for success rate computation
    rsi_window: int = 20               # Window for RSI smoothing

    # Red-flag observation thresholds (for LOGGING only, NOT enforcement)
    cdi_010_observation_enabled: bool = True
    cdi_007_streak_threshold: int = 10      # Log warning after N consecutive
    rsi_collapse_threshold: float = 0.2     # Log warning when ρ < threshold
    omega_exit_threshold: int = 100         # Log warning after N cycles outside Ω
    block_rate_threshold: float = 0.6       # Log warning when β > threshold
    divergence_streak_threshold: int = 20   # Log warning after N consecutive

    # SHADOW MODE: These are observation-only, never enforced
    shadow_mode: bool = True           # Must always be True in P3
```

### 2.3 Output Contract

```python
@dataclass
class FirstLightResult:
    """Result of First-Light shadow experiment."""

    # Run metadata
    run_id: str
    config: FirstLightConfig
    start_time: str                    # ISO 8601
    end_time: str                      # ISO 8601

    # Cycle summary
    total_cycles_run: int
    cycles_completed: int

    # Success metrics (U2)
    u2_success_rate_final: Optional[float]
    u2_success_rate_trajectory: List[float]
    delta_p_success: Optional[float]

    # Abstention metrics (RFL)
    rfl_abstention_rate_final: Optional[float]
    rfl_abstention_trajectory: List[float]
    delta_p_abstention: Optional[float]

    # Stability metrics
    mean_rsi: float
    rsi_trajectory: List[float]

    # Safe region metrics
    omega_occupancy: float
    omega_occupancy_trajectory: List[float]

    # HARD mode metrics
    hard_ok_rate: float
    hard_ok_trajectory: List[float]

    # Red-flag observations (LOGGED, not enforced)
    red_flag_summary: RedFlagSummary

    # Output paths
    cycles_log_path: str
    red_flags_log_path: str
    metrics_log_path: str
    summary_path: str
```

### 2.4 API Methods

```python
class FirstLightShadowRunner:
    """
    First-Light shadow experiment runner.

    SHADOW MODE CONTRACT:
    - NEVER modifies governance decisions
    - NEVER enforces abort conditions
    - All outputs are observational only
    - Red-flags are LOGGED, not ACTED upon
    """

    def __init__(self, config: FirstLightConfig) -> None:
        """Initialize runner with validated config."""

    def run(self) -> FirstLightResult:
        """Execute shadow experiment (SHADOW MODE: observation only)."""

    def run_cycles(self, n: int) -> Iterator[CycleObservation]:
        """Run N cycles and yield observations."""

    def get_current_metrics(self) -> MetricsSnapshot:
        """Get current metrics window snapshot."""

    def get_red_flag_status(self) -> RedFlagStatus:
        """Get current red-flag observation status (observational only)."""

    def finalize(self) -> FirstLightResult:
        """Finalize experiment and write summary."""
```

---

## 3. Red-Flag Observation Layer

### 3.1 Design Principles

The Red-Flag Observation Layer **LOGS** anomalous conditions but **NEVER ENFORCES** them.

```
┌─────────────────────────────────────────────────────────────────┐
│                  Red-Flag Observation Layer                      │
│                                                                  │
│  SHADOW MODE CONTRACT:                                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  ✗ NO enforcement      ✗ NO aborts       ✗ NO control flow │ │
│  │  ✓ Logging only        ✓ Metrics only    ✓ Observation only│ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Observed Conditions:                                           │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐ │
│  │   CDI-010    │   CDI-007    │ RSI Collapse │  Ω Exit      │ │
│  │  (any act.)  │  (streak)    │  (ρ < 0.2)   │ (duration)   │ │
│  └──────────────┴──────────────┴──────────────┴──────────────┘ │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐ │
│  │ Block Rate   │  Threshold   │  Governance  │  HARD Mode   │ │
│  │  (β > 0.6)   │   Drift      │  Divergence  │  Failure     │ │
│  └──────────────┴──────────────┴──────────────┴──────────────┘ │
│                                                                  │
│  Output: red_flags.jsonl (observation log only)                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Red-Flag Definitions

| Red-Flag | Condition | Threshold | Log Level | Action |
|----------|-----------|-----------|-----------|--------|
| `CDI-010` | Fixed-Point Multiplicity | Any activation | CRITICAL | **LOG ONLY** |
| `CDI-007` | Exception Exhaustion | > 10 consecutive | WARNING | **LOG ONLY** |
| `RSI_COLLAPSE` | ρ < ρ_min | ρ < 0.2 for > 10 cycles | WARNING | **LOG ONLY** |
| `OMEGA_EXIT` | State outside Ω | > 100 consecutive cycles | WARNING | **LOG ONLY** |
| `BLOCK_RATE_EXPLOSION` | β > β_max | β > 0.6 for > 20 cycles | WARNING | **LOG ONLY** |
| `THRESHOLD_DRIFT` | \|τ - τ_0\| > ε | > 0.05 drift for > 50 cycles | INFO | **LOG ONLY** |
| `GOVERNANCE_DIVERGENCE` | real ≠ sim | > 20 consecutive CRITICAL | WARNING | **LOG ONLY** |
| `HARD_FAIL` | HARD_OK = False | > 50 consecutive cycles | WARNING | **LOG ONLY** |

### 3.3 Data Structures

```python
class RedFlagType(Enum):
    """Types of red-flag observations."""
    CDI_010 = "CDI-010"
    CDI_007 = "CDI-007"
    RSI_COLLAPSE = "RSI_COLLAPSE"
    OMEGA_EXIT = "OMEGA_EXIT"
    BLOCK_RATE_EXPLOSION = "BLOCK_RATE_EXPLOSION"
    THRESHOLD_DRIFT = "THRESHOLD_DRIFT"
    GOVERNANCE_DIVERGENCE = "GOVERNANCE_DIVERGENCE"
    HARD_FAIL = "HARD_FAIL"


class RedFlagSeverity(Enum):
    """Severity levels for red-flag observations."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class RedFlagObservation:
    """A single red-flag observation (LOGGED, not enforced)."""
    cycle: int
    timestamp: str
    flag_type: RedFlagType
    severity: RedFlagSeverity
    observed_value: float
    threshold: float
    consecutive_cycles: int
    state_snapshot: Dict[str, Any]
    action_taken: str = "LOGGED_ONLY"  # Always "LOGGED_ONLY" in P3


@dataclass
class RedFlagSummary:
    """Summary of all red-flag observations in a run."""
    total_observations: int
    observations_by_type: Dict[str, int]
    observations_by_severity: Dict[str, int]
    max_cdi_007_streak: int
    max_rsi_collapse_streak: int
    max_omega_exit_streak: int
    max_block_rate_streak: int
    max_divergence_streak: int
    max_hard_fail_streak: int
    cdi_010_activations: int
    hypothetical_abort_cycles: List[int]
    hypothetical_abort_reasons: List[str]
```

### 3.4 Observer API

```python
class RedFlagObserver:
    """
    Observes red-flag conditions without enforcing them.

    SHADOW MODE CONTRACT:
    - observe() NEVER returns an abort signal
    - observe() NEVER modifies control flow
    - hypothetical_should_abort() is for analysis ONLY
    """

    def observe(self, cycle: int, state: USLAState, hard_ok: bool,
                governance_aligned: bool) -> List[RedFlagObservation]:
        """Observe current state for red-flag conditions (LOGGING only)."""

    def hypothetical_should_abort(self) -> Tuple[bool, Optional[str]]:
        """Check if abort WOULD be triggered (analysis only, NEVER enforced)."""

    def get_summary(self) -> RedFlagSummary:
        """Get summary of all observations."""

    def reset(self) -> None:
        """Reset observer state."""
```

---

## 4. Δp Computation Contract

### 4.1 Overview

Δp metrics measure learning curves in SHADOW mode. All computations are read-only and observational.

```
┌─────────────────────────────────────────────────────────────────┐
│                      Δp Computation                              │
│                                                                  │
│  SHADOW MODE: All metrics are computed from observed data.      │
│  No governance decisions are influenced by these values.        │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Δp_success (U2): Slope of success rate over cycles       │  │
│  │  Target: Δp > 0 (positive learning)                       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Δp_abstention (RFL): Slope of abstention rate            │  │
│  │  Target: Δp < 0 (decreasing abstention)                   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Ω Occupancy: Fraction of cycles in safe region Ω        │  │
│  │  Target: Ω_occupancy ≥ 0.90                               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  HARD-OK Curve: Fraction with HARD mode OK                │  │
│  │  Target: HARD_OK ≥ 0.80                                   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Structures

```python
@dataclass
class DeltaPMetrics:
    """Learning curve metrics computed in SHADOW mode (observational only)."""

    # U2 metrics
    delta_p_success: Optional[float]
    success_rate_trajectory: List[float]
    success_rate_final: Optional[float]

    # RFL metrics
    delta_p_abstention: Optional[float]
    abstention_trajectory: List[float]
    abstention_rate_final: Optional[float]

    # Safe region metrics
    omega_occupancy: float
    omega_occupancy_trajectory: List[float]

    # HARD mode metrics
    hard_ok_rate: float
    hard_ok_trajectory: List[float]

    # RSI metrics
    mean_rsi: float
    rsi_trajectory: List[float]

    # Window configuration
    window_size: int
    total_windows: int

    def meets_success_criteria(self) -> Dict[str, bool]:
        """Check success criteria (for LOGGING only, not control flow)."""
```

### 4.3 Linear Regression for Δp

```python
def compute_slope(values: List[float]) -> Optional[float]:
    """
    Compute slope via simple linear regression.

    Uses least squares: slope = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)

    Returns:
        Slope, or None if insufficient data
    """
```

---

## 5. JSONL Log Schemas

### 5.1 Cycle Log (`cycles.jsonl`)

```json
{
  "schema": "first-light-cycle/1.0.0",
  "cycle": 42,
  "timestamp": "2025-12-09T12:00:00.000000+00:00",
  "mode": "SHADOW",

  "runner": {
    "type": "u2",
    "slice": "arithmetic_simple",
    "success": true,
    "depth": 4
  },

  "usla_state": {
    "H": 0.75,
    "D": 5,
    "D_dot": 0.5,
    "B": 2.0,
    "S": 0.1,
    "C": "CONVERGING",
    "rho": 0.85,
    "tau": 0.21,
    "J": 2.5,
    "W": false,
    "beta": 0.05,
    "kappa": 0.8,
    "nu": 0.001,
    "delta": 0,
    "Gamma": 0.88
  },

  "governance": {
    "real_blocked": false,
    "sim_blocked": false,
    "aligned": true
  },

  "metrics": {
    "hard_ok": true,
    "in_omega": true,
    "window_success_rate": 0.78,
    "window_rsi": 0.83
  }
}
```

### 5.2 Red-Flag Log (`red_flags.jsonl`)

```json
{
  "schema": "first-light-red-flag/1.0.0",
  "cycle": 142,
  "timestamp": "2025-12-09T12:01:42.000000+00:00",
  "mode": "SHADOW",

  "flag": {
    "type": "RSI_COLLAPSE",
    "severity": "WARNING",
    "observed_value": 0.18,
    "threshold": 0.20,
    "consecutive_cycles": 12
  },

  "action": "LOGGED_ONLY",

  "context": {
    "state_H": 0.45,
    "state_rho": 0.18,
    "state_beta": 0.42,
    "in_omega": false,
    "hard_ok": false
  },

  "hypothetical": {
    "would_abort": false,
    "reason": null
  }
}
```

### 5.3 Metrics Log (`metrics.jsonl`)

```json
{
  "schema": "first-light-metrics/1.0.0",
  "window_index": 8,
  "window_start_cycle": 400,
  "window_end_cycle": 449,
  "timestamp": "2025-12-09T12:08:00.000000+00:00",
  "mode": "SHADOW",

  "success_metrics": {
    "window_success_rate": 0.82,
    "cumulative_success_rate": 0.76,
    "delta_p_success": 0.0012
  },

  "abstention_metrics": {
    "window_abstention_rate": 0.12,
    "cumulative_abstention_rate": 0.14,
    "delta_p_abstention": -0.0008
  },

  "stability_metrics": {
    "window_mean_rsi": 0.84,
    "cumulative_mean_rsi": 0.81,
    "rsi_variance": 0.012
  },

  "safe_region_metrics": {
    "window_omega_occupancy": 0.94,
    "cumulative_omega_occupancy": 0.91
  },

  "hard_mode_metrics": {
    "window_hard_ok_rate": 0.88,
    "cumulative_hard_ok_rate": 0.84
  },

  "red_flag_counts": {
    "CDI-010": 0,
    "CDI-007": 2,
    "RSI_COLLAPSE": 1,
    "OMEGA_EXIT": 0
  }
}
```

### 5.4 Summary (`summary.json`)

```json
{
  "schema": "first-light-summary/1.0.0",
  "run_id": "fl_20251209_120000_abc123",
  "mode": "SHADOW",

  "config": {
    "slice_name": "arithmetic_simple",
    "runner_type": "u2",
    "total_cycles": 1000,
    "tau_0": 0.20
  },

  "execution": {
    "start_time": "2025-12-09T12:00:00.000000+00:00",
    "end_time": "2025-12-09T12:16:40.000000+00:00",
    "duration_seconds": 1000.0,
    "cycles_completed": 1000
  },

  "success_criteria": {
    "u2_success_rate_75": {"target": 0.75, "actual": 0.78, "passed": true},
    "delta_p_positive": {"target": 0.0, "actual": 0.0015, "passed": true},
    "mean_rsi_60": {"target": 0.60, "actual": 0.81, "passed": true},
    "omega_occupancy_90": {"target": 0.90, "actual": 0.92, "passed": true},
    "cdi_010_zero": {"target": 0, "actual": 0, "passed": true},
    "cdi_007_under_50": {"target": 50, "actual": 12, "passed": true},
    "hard_ok_80": {"target": 0.80, "actual": 0.86, "passed": true}
  },

  "red_flag_summary": {
    "total_observations": 15,
    "hypothetical_aborts": 0
  }
}
```

---

## 6. Example 50-Cycle Shadow Test Output

### 6.1 Scenario

A 50-cycle shadow test with:
- Slice: `arithmetic_simple` (U2)
- τ₀ = 0.20
- One RSI dip at cycles 25-30

### 6.2 Sample Cycle Entries

```jsonl
{"schema":"first-light-cycle/1.0.0","cycle":1,"mode":"SHADOW","runner":{"type":"u2","slice":"arithmetic_simple","success":true},"usla_state":{"H":1.0,"rho":1.0,"tau":0.20},"governance":{"aligned":true},"metrics":{"hard_ok":true,"in_omega":true}}
{"schema":"first-light-cycle/1.0.0","cycle":25,"mode":"SHADOW","runner":{"type":"u2","slice":"arithmetic_simple","success":false},"usla_state":{"H":0.52,"rho":0.45,"tau":0.22},"governance":{"aligned":true},"metrics":{"hard_ok":false,"in_omega":false}}
{"schema":"first-light-cycle/1.0.0","cycle":50,"mode":"SHADOW","runner":{"type":"u2","slice":"arithmetic_simple","success":true},"usla_state":{"H":0.78,"rho":0.82,"tau":0.21},"governance":{"aligned":true},"metrics":{"hard_ok":true,"in_omega":true}}
```

### 6.3 Sample Red-Flag Entry

```jsonl
{"schema":"first-light-red-flag/1.0.0","cycle":28,"mode":"SHADOW","flag":{"type":"RSI_COLLAPSE","severity":"WARNING","observed_value":0.35,"threshold":0.40,"consecutive_cycles":4},"action":"LOGGED_ONLY","hypothetical":{"would_abort":false}}
```

---

## 7. Safety Boundaries

### 7.1 SHADOW MODE Invariants

| Invariant | Enforcement | Violation Handling |
|-----------|-------------|-------------------|
| No governance modification | Code review + tests | CI failure |
| No abort control flow | Code review + tests | CI failure |
| All outputs observational | Docstrings + schema | Documentation audit |
| Red-flags logged only | `action: LOGGED_ONLY` | Schema validation |
| Hypotheticals clearly marked | Field naming | Schema validation |

### 7.2 Boundary Checks in Code

```python
class FirstLightShadowRunner:
    def __init__(self, config: FirstLightConfig) -> None:
        # SHADOW MODE: Enforce at initialization
        if not config.shadow_mode:
            raise ValueError(
                "SHADOW MODE VIOLATION: FirstLightShadowRunner requires "
                "shadow_mode=True. Active mode is not authorized in Phase X P3."
            )
```

### 7.3 Schema Validation

All JSONL entries must include:
- `"mode": "SHADOW"` — Enforced by schema
- `"action": "LOGGED_ONLY"` — For red-flag entries
- `"hypothetical"` prefix — For would-have-aborted analysis

---

## 8. CI Requirements

### 8.1 Test Categories

| Category | Description | Blocking |
|----------|-------------|----------|
| `test_shadow_mode_enforced` | Verify SHADOW mode checks | Yes |
| `test_config_validation` | Config validation | Yes |
| `test_jsonl_schema_compliance` | Log schema validation | Yes |
| `test_red_flag_logging_only` | No enforcement verification | Yes |
| `test_delta_p_computation` | Δp calculation accuracy | Yes |
| `test_50_cycle_smoke` | 50-cycle shadow run | Yes |

### 8.2 Proposed Workflow Extension

```yaml
# Future addition to .github/workflows/usla-shadow-gate.yml
jobs:
  first-light-preparation:
    name: First-Light P3 Validation
    runs-on: ubuntu-latest
    env:
      USLA_SHADOW_ENABLED: 'true'
    steps:
      - name: Validate First-Light Config Schema
        run: uv run pytest tests/first_light/test_config_validation.py -v
      - name: Verify SHADOW Mode Enforcement
        run: uv run pytest tests/first_light/test_shadow_mode_enforced.py -v
```

---

## 9. Implementation Plan

### 9.1 File Structure

```
backend/
├── topology/
│   ├── first_light/
│   │   ├── __init__.py
│   │   ├── config.py              # FirstLightConfig, FirstLightResult
│   │   ├── runner.py              # FirstLightShadowRunner (STUB)
│   │   ├── red_flag_observer.py   # RedFlagObserver (STUB)
│   │   ├── delta_p_computer.py    # DeltaPComputer (STUB)
│   │   ├── metrics_window.py      # MetricsWindow (STUB)
│   │   └── schemas.py             # JSONL schema definitions
│   └── ...existing files...

tests/
├── first_light/
│   ├── __init__.py
│   └── test_first_light_stubs.py  # Import smoke tests only

docs/
└── system_law/
    └── Phase_X_P3_Spec.md          # This document
```

### 9.2 Data Structures Summary

| Structure | File | Purpose |
|-----------|------|---------|
| `FirstLightConfig` | `config.py` | Experiment configuration |
| `FirstLightResult` | `config.py` | Experiment results |
| `RedFlagType` | `red_flag_observer.py` | Flag type enum |
| `RedFlagSeverity` | `red_flag_observer.py` | Severity enum |
| `RedFlagObservation` | `red_flag_observer.py` | Single observation |
| `RedFlagSummary` | `red_flag_observer.py` | Run summary |
| `DeltaPMetrics` | `delta_p_computer.py` | Learning metrics |
| `MetricsWindow` | `metrics_window.py` | Windowed metrics |

### 9.3 Function Summary

| Function | File | Status |
|----------|------|--------|
| `FirstLightConfig.validate()` | `config.py` | STUB |

### 9.4 Harness Entry Points

> **Compatibility Notice**: `scripts/first_light_p3_harness.py` is retained only as a wrapper around `scripts/usla_first_light_harness.py` for legacy tooling. All new orchestration, demos, and CI hooks must call `scripts/usla_first_light_harness.py` directly so that the real harness remains the single source of truth.
| `FirstLightShadowRunner.run()` | `runner.py` | STUB |
| `RedFlagObserver.observe()` | `red_flag_observer.py` | STUB |
| `DeltaPComputer.compute()` | `delta_p_computer.py` | STUB |
| `compute_slope()` | `delta_p_computer.py` | STUB |

---

## 10. Authorization Gates

### 10.1 Current Authorization (P3 Design Freeze)

| Capability | Authorized | Notes |
|------------|------------|-------|
| Design specification | ✅ Yes | This document |
| Code skeletons/stubs | ✅ Yes | No real logic |
| Import smoke tests | ✅ Yes | Minimal validation |
| JSONL schema docs | ✅ Yes | Documentation only |

### 10.2 NOT Authorized (Requires Future Phase)

| Capability | Phase Required |
|------------|----------------|
| Working FirstLightShadowRunner | Explicit P3 execution auth |
| Real 50/1000-cycle experiments | Explicit P3 execution auth |
| Abort logic enforcement | Phase XI |
| Governance integration | Phase XI |
| HARD mode activation | Phase XI |

---

## 11. Summary

This Phase X P3 specification provides:

1. **Complete API design** for FirstLightShadowRunner
2. **Red-Flag Observation Layer** (logging only, never enforcement)
3. **Δp computation contracts** for learning curve analysis
4. **JSONL log schemas** for all output types
5. **Example 50-cycle output** demonstrating expected data
6. **Safety boundaries** ensuring SHADOW MODE compliance
7. **CI requirements** for validation
8. **Implementation plan** with files, functions, tests

**SHADOW MODE CONTRACT MAINTAINED**: All designs are observational only. No governance modifications. No abort enforcement.

**Implementation requires explicit future authorization.**
