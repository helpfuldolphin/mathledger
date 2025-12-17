# Phase X P4 Specification: Real Runner Shadow Coupling

---

> **PHASE X P4 — DESIGN DOCUMENT — IMPLEMENTATION NOT AUTHORIZED**
>
> This document defines the interface and data contracts for Phase X P4.
> All skeleton code MUST raise `NotImplementedError` until explicit
> implementation authorization is granted.
>
> **SHADOW MODE ONLY. NO GOVERNANCE MODIFICATION. NO ABORT ENFORCEMENT.**

---

**Status**: Design Freeze (Stubs Only)
**Phase**: X P4 (SHADOW MODE ONLY)
**Version**: 1.0.0
**Date**: 2025-12-09

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Strategic Rationale (STRATCOM Directive)](#2-strategic-rationale-stratcom-directive)
3. [Absolute Invariants](#3-absolute-invariants)
4. [Architecture](#4-architecture)
5. [Interface Specifications](#5-interface-specifications)
6. [Data Structures](#6-data-structures)
7. [Schema Specifications](#7-schema-specifications)
8. [Experiment Flow](#8-experiment-flow)
9. [Test Plan](#9-test-plan)
10. [Non-Goals](#10-non-goals)
11. [Future Bridge to P5](#11-future-bridge-to-p5)
12. [Implementation Skeletons](#12-implementation-skeletons)
13. [Authorization Gates](#13-authorization-gates)

---

## 1. Executive Summary

Phase X P4 extends the First-Light shadow experiment architecture from P3's synthetic data to **real runner telemetry coupling**. The P3 runner (`FirstLightShadowRunner`) observes synthetic state generated internally. P4 introduces a read-only adapter that observes **real telemetry** from actual U2Runner and RFLRunner executions, while maintaining the absolute constraint that no feedback flows from shadow to real execution.

### P3 vs P4 Comparison

| Aspect | P3 (Synthetic) | P4 (Real Coupling) |
|--------|----------------|-------------------|
| Data source | Internal synthetic generator | Real runner telemetry |
| USLAIntegration | Not used | Read-only observation via adapter |
| Divergence analysis | N/A | Real vs Twin comparison |
| Governance modification | PROHIBITED | PROHIBITED |
| Abort enforcement | PROHIBITED | PROHIBITED |
| Observer effect | None | Must be eliminated |

### Key Deliverables

1. **TelemetryProviderInterface** — Abstract interface for telemetry sources
2. **USLAIntegrationAdapter** — Read-only adapter for real runner observation
3. **FirstLightShadowRunnerP4** — Extended runner consuming real telemetry
4. **DivergenceAnalyzer** — Real vs Twin trajectory comparison
5. **P4 Schemas** — JSONL formats for real cycles, twin cycles, divergence snapshots

---

## 2. Strategic Rationale (STRATCOM Directive)

### The Scaling Problem

Contemporary AI development exhibits a dangerous assumption: that sufficient scaling will spontaneously produce reliable cognitive governance. This is categorically false. Intelligence and stability are orthogonal dimensions. A system may exhibit arbitrarily high capability while possessing arbitrarily poor self-regulation.

### Why Substrate-Level Governance Cannot Emerge

1. **Optimization pressure doesn't select for stability** — Systems optimized for capability metrics will develop increasingly sophisticated goal-pursuit without developing constraints on that pursuit.

2. **Self-modeling doesn't imply self-governance** — A system may perfectly understand its own operation while lacking mechanisms to constrain that operation.

3. **Scale amplifies misalignment** — More capable systems can pursue misaligned objectives more effectively, not less.

### The Retrofit Impossibility

Attempting to add governance constraints to post-hoc systems faces fundamental obstacles:

- **Structural Incompatibility**: Control surfaces must be architectural features, not patches
- **Optimization Circumvention**: Systems will route around constraints if optimization pressure permits
- **Verification Impossibility**: Cannot verify a system constrains itself without independent verification infrastructure

### Phase 0 Doctrine

Substrate-level governance must be established **before** scaling creates systems too capable to safely constrain. This requires:

1. **Verifiable derivation paths** — Every capability claim must trace to axiomatic foundations
2. **Independent verification** — External systems (Lean, Z3) validate claims without trusting the claiming system
3. **Cryptographic commitment** — Irreversible records prevent post-hoc modification
4. **Shadow observation** — Measure system behavior without influencing it

### P4's Role in Phase 0

P4 represents a critical transition: moving from **synthetic observation** (P3) to **real system observation** (P4) while maintaining the shadow guarantee. This establishes:

- Proof that observation infrastructure can couple to real execution
- Verification that no feedback leaks from shadow to real paths
- Foundation for eventual active governance (P5+, not authorized)

### MathLedger as Verification Substrate

MathLedger provides the truth anchor for P4:

- Every derivation claim → Lean verification → cryptographic attestation
- Shadow observations recorded as ledger-compatible JSONL
- Divergence events create auditable evidence chains

**P4 demonstrates that we can observe real system behavior through the same infrastructure that will eventually govern it, without that observation altering the behavior.**

---

## 3. Absolute Invariants

These invariants are **non-negotiable** and apply to all P4 code:

### 3.1 Invariant Table

| ID | Invariant | Enforcement | Violation = |
|----|-----------|-------------|-------------|
| INV-01 | **Shadow Mode Only** | All P4 code operates in observation-only mode | CRITICAL BUG |
| INV-02 | **Read-Only Coupling** | Adapter NEVER writes to USLAIntegration | CRITICAL BUG |
| INV-03 | **No Governance Modification** | Real runner decisions are never altered | CRITICAL BUG |
| INV-04 | **No Abort Enforcement** | Red-flags are logged, never acted upon | CRITICAL BUG |
| INV-05 | **USLAIntegration Bridge Only** | All real data flows through adapter interface | CRITICAL BUG |
| INV-06 | **Explicit Mode Declaration** | All logs include `"mode": "SHADOW"` | Schema Error |
| INV-07 | **Observer Effect Avoidance** | Shadow observation CANNOT affect real execution | CRITICAL BUG |

### 3.2 Enforcement Mechanisms

```python
class FirstLightShadowRunnerP4:
    def __init__(self, config: FirstLightConfigP4) -> None:
        # INV-01: Shadow mode enforcement at initialization
        if not config.shadow_mode:
            raise ValueError(
                "SHADOW MODE VIOLATION: FirstLightShadowRunnerP4 requires "
                "shadow_mode=True. Active mode is not authorized in Phase X P4."
            )

        # INV-02, INV-05: Adapter must be read-only
        if config.telemetry_adapter is not None:
            if not isinstance(config.telemetry_adapter, TelemetryProviderInterface):
                raise TypeError(
                    "P4 INVARIANT VIOLATION: telemetry_adapter must implement "
                    "TelemetryProviderInterface (read-only)"
                )
```

---

## 4. Architecture

### 4.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Phase X P4 Architecture                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                    REAL EXECUTION PATH (UNTOUCHED)                        │   │
│  │                                                                           │   │
│  │   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐             │   │
│  │   │  U2Runner   │      │  RFLRunner  │      │ Governance  │             │   │
│  │   │             │      │             │      │  Layer      │             │   │
│  │   └──────┬──────┘      └──────┬──────┘      └──────┬──────┘             │   │
│  │          │                    │                    │                     │   │
│  │          ▼                    ▼                    ▼                     │   │
│  │   ┌──────────────────────────────────────────────────────────────┐      │   │
│  │   │                   USLAIntegration                             │      │   │
│  │   │              (existing production path)                       │      │   │
│  │   └─────────────────────────┬────────────────────────────────────┘      │   │
│  │                             │                                           │   │
│  └─────────────────────────────┼───────────────────────────────────────────┘   │
│                                │                                               │
│                     ╔══════════╧══════════╗                                    │
│                     ║   READ-ONLY FENCE   ║                                    │
│                     ║  (No writes allowed) ║                                    │
│                     ╚══════════╤══════════╝                                    │
│                                │                                               │
│  ┌─────────────────────────────┼───────────────────────────────────────────┐   │
│  │                    SHADOW OBSERVATION PATH                               │   │
│  │                                                                          │   │
│  │                    ┌────────▼────────┐                                  │   │
│  │                    │ USLAIntegration │                                  │   │
│  │                    │    Adapter      │                                  │   │
│  │                    │  (READ-ONLY)    │                                  │   │
│  │                    └────────┬────────┘                                  │   │
│  │                             │                                           │   │
│  │            ┌────────────────┼────────────────┐                         │   │
│  │            ▼                ▼                ▼                          │   │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │   │
│  │   │ RealCycle    │  │ TwinRunner   │  │ Divergence   │                │   │
│  │   │ Observer     │  │ (Shadow Sim) │  │ Analyzer     │                │   │
│  │   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                │   │
│  │          │                 │                 │                         │   │
│  │          ▼                 ▼                 ▼                         │   │
│  │   ┌──────────────────────────────────────────────────────────────┐    │   │
│  │   │               FirstLightShadowRunnerP4                        │    │   │
│  │   │                                                               │    │   │
│  │   │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │    │   │
│  │   │  │ RedFlagObserver │  │ DeltaPComputer  │  │ MetricsAcc  │  │    │   │
│  │   │  │ (logging only)  │  │ (read-only)     │  │ (read-only) │  │    │   │
│  │   │  └─────────────────┘  └─────────────────┘  └─────────────┘  │    │   │
│  │   └───────────────────────────────┬──────────────────────────────┘    │   │
│  │                                   │                                   │   │
│  └───────────────────────────────────┼───────────────────────────────────┘   │
│                                      │                                       │
│                                      ▼                                       │
│   ┌────────────────────────────────────────────────────────────────────────┐ │
│   │                        Output Artifacts                                 │ │
│   │                                                                         │ │
│   │   results/first_light_p4/{run_id}/                                     │ │
│   │   ├── real_cycles.jsonl      # Observed real runner telemetry          │ │
│   │   ├── twin_cycles.jsonl      # Shadow twin predictions                 │ │
│   │   ├── divergence.jsonl       # Real vs twin comparison                 │ │
│   │   ├── red_flags.jsonl        # Anomaly observations                    │ │
│   │   ├── metrics.jsonl          # Windowed metrics                        │ │
│   │   └── summary.json           # Run summary                             │ │
│   └────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

All Phase X First Light JSONL artifacts (synthetic_raw, real_cycles, twin_predictions, divergence_log, metrics, etc.) are emitted via the shared `backend.logging.jsonl_writer.JsonlWriter` helper to keep ordering deterministic and flush semantics uniform; rotation/rollover is deferred to **Phase Y**, where size-based thresholds will trigger new suffixed segments without changing payload schemas. Determinism comparisons ignore timestamp fields but require stable numeric rounding (four decimal precision) so that replayed runs match exactly apart from wall-clock jitter.

### 4.2 Data Flow

```
Real Runner Execution
         │
         ▼
   USLAIntegration
   (production path)
         │
    ┌────┴────┐
    │  FENCE  │ ← No data flows BACK through this fence
    └────┬────┘
         │ (read-only snapshot)
         ▼
USLAIntegrationAdapter
         │
    ┌────┴────────────────────────┐
    │                             │
    ▼                             ▼
RealCycleObserver            TwinRunner
    │                             │
    │                             ▼
    │                      TwinCycleObservation
    │                             │
    └──────────┬──────────────────┘
               │
               ▼
       DivergenceAnalyzer
               │
               ▼
       DivergenceSnapshot
               │
               ▼
      FirstLightShadowRunnerP4
               │
    ┌──────────┼──────────────────┐
    │          │                  │
    ▼          ▼                  ▼
RedFlag    DeltaP           MetricsAccum
Observer   Computer         ulator
    │          │                  │
    └──────────┴──────────────────┘
               │
               ▼
         JSONL Output
```

---

## 5. Interface Specifications

### 5.1 TelemetrySnapshot (Frozen Dataclass)

```python
from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime

@dataclass(frozen=True)
class TelemetrySnapshot:
    """
    Immutable snapshot of runner telemetry.

    SHADOW MODE: This is a READ-ONLY capture of real runner state.
    No methods modify any external state.
    """

    # Cycle identification
    cycle: int
    timestamp: datetime
    runner_type: str  # "u2" or "rfl"
    slice_name: str

    # Runner outcome
    success: bool
    depth: Optional[int]
    proof_hash: Optional[str]

    # USLA state
    H: float           # Health metric
    rho: float         # RSI (Running Stability Index)
    tau: float         # Current threshold
    beta: float        # Block rate
    in_omega: bool     # Safe region membership

    # Governance state
    real_blocked: bool
    governance_aligned: bool

    # HARD mode
    hard_ok: bool

    # Abstention (RFL only)
    abstained: Optional[bool]
    abstention_reason: Optional[str]

    # Extended metrics
    extended: Dict[str, Any]
```

### 5.2 TelemetryProviderInterface

```python
from abc import ABC, abstractmethod
from typing import Iterator, Optional

class TelemetryProviderInterface(ABC):
    """
    Abstract interface for telemetry providers.

    SHADOW MODE CONTRACT:
    - All methods are READ-ONLY
    - No method modifies external state
    - No method influences real runner execution
    - Implementations must be observation-only
    """

    @abstractmethod
    def get_current_snapshot(self) -> Optional[TelemetrySnapshot]:
        """
        Get current telemetry snapshot (READ-ONLY).

        Returns:
            Current snapshot, or None if no data available
        """
        raise NotImplementedError("P4 implementation not yet activated")

    @abstractmethod
    def get_historical_snapshots(
        self, start_cycle: int, end_cycle: int
    ) -> Iterator[TelemetrySnapshot]:
        """
        Get historical snapshots in range (READ-ONLY).

        Args:
            start_cycle: First cycle (inclusive)
            end_cycle: Last cycle (inclusive)

        Yields:
            TelemetrySnapshot for each cycle in range
        """
        raise NotImplementedError("P4 implementation not yet activated")

    @abstractmethod
    def is_available(self) -> bool:
        """Check if telemetry source is available (READ-ONLY)."""
        raise NotImplementedError("P4 implementation not yet activated")

    @abstractmethod
    def get_runner_type(self) -> str:
        """Get runner type being observed (READ-ONLY)."""
        raise NotImplementedError("P4 implementation not yet activated")
```

### 5.3 USLAIntegrationAdapter

```python
class USLAIntegrationAdapter(TelemetryProviderInterface):
    """
    Read-only adapter for USLAIntegration telemetry.

    SHADOW MODE CONTRACT:
    - NEVER writes to USLAIntegration
    - NEVER modifies governance state
    - NEVER influences runner execution
    - All access is observation-only via exposed metrics

    This adapter reads from USLAIntegration's public metrics interfaces
    without calling any methods that could modify state.
    """

    def __init__(
        self,
        integration_ref: "USLAIntegration",  # Type hint only
        runner_type: str,
    ) -> None:
        """
        Initialize read-only adapter.

        Args:
            integration_ref: Reference to USLAIntegration instance
            runner_type: "u2" or "rfl"

        SHADOW MODE: This constructor validates read-only contract.
        """
        raise NotImplementedError("P4 implementation not yet activated")
```

---

## 6. Data Structures

### 6.1 RealCycleObservation

```python
@dataclass
class RealCycleObservation:
    """
    Observation of a real runner cycle.

    SHADOW MODE: This captures what actually happened.
    It does NOT influence what happens next.
    """

    # Source identification
    source: str = "REAL_RUNNER"

    # Cycle data
    cycle: int = 0
    timestamp: str = ""

    # Runner outcome
    runner_type: str = ""
    slice_name: str = ""
    success: bool = False
    depth: Optional[int] = None

    # USLA state snapshot
    H: float = 0.0
    rho: float = 0.0
    tau: float = 0.0
    beta: float = 0.0
    in_omega: bool = False

    # Governance
    real_blocked: bool = False
    governance_aligned: bool = True

    # HARD mode
    hard_ok: bool = True

    # Abstention
    abstained: bool = False
    abstention_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSONL serialization."""
        raise NotImplementedError("P4 implementation not yet activated")
```

### 6.2 TwinCycleObservation

```python
@dataclass
class TwinCycleObservation:
    """
    Shadow twin prediction for a cycle.

    SHADOW MODE: This is what the twin PREDICTED would happen,
    computed without influencing the real execution.
    """

    # Source identification
    source: str = "SHADOW_TWIN"

    # Corresponding real cycle
    real_cycle: int = 0
    timestamp: str = ""

    # Twin predictions
    predicted_success: bool = False
    predicted_blocked: bool = False
    predicted_in_omega: bool = False
    predicted_hard_ok: bool = True

    # Twin state
    twin_H: float = 0.0
    twin_rho: float = 0.0
    twin_tau: float = 0.0
    twin_beta: float = 0.0

    # Confidence metrics
    prediction_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSONL serialization."""
        raise NotImplementedError("P4 implementation not yet activated")
```

### 6.3 DivergenceSnapshot

```python
@dataclass
class DivergenceSnapshot:
    """
    Comparison between real and twin observations.

    SHADOW MODE: This analysis is for logging only.
    Divergences do NOT trigger any remediation.
    """

    cycle: int = 0
    timestamp: str = ""

    # Divergence flags
    success_diverged: bool = False
    blocked_diverged: bool = False
    omega_diverged: bool = False
    hard_ok_diverged: bool = False

    # Magnitude metrics
    H_delta: float = 0.0
    rho_delta: float = 0.0
    tau_delta: float = 0.0
    beta_delta: float = 0.0

    # Classification
    divergence_severity: str = "NONE"  # NONE, MINOR, MODERATE, SEVERE
    divergence_type: str = "NONE"      # NONE, STATE, OUTCOME, BOTH

    # Analysis
    consecutive_divergences: int = 0
    divergence_streak_start: Optional[int] = None

    # Action taken
    action: str = "LOGGED_ONLY"  # Always "LOGGED_ONLY" in P4

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSONL serialization."""
        raise NotImplementedError("P4 implementation not yet activated")
```

### 6.4 FirstLightConfigP4

```python
@dataclass
class FirstLightConfigP4:
    """
    Configuration for P4 shadow experiment with real coupling.

    SHADOW MODE: shadow_mode must always be True.
    """

    # Slice and runner selection
    slice_name: str = "arithmetic_simple"
    runner_type: str = "u2"  # "u2" or "rfl"

    # Run parameters
    total_cycles: int = 1000
    tau_0: float = 0.20

    # Telemetry adapter (P4-specific)
    telemetry_adapter: Optional[TelemetryProviderInterface] = None

    # Logging configuration
    log_dir: str = "results/first_light_p4"
    run_id: Optional[str] = None
    log_every_n_cycles: int = 1

    # Windows
    success_window: int = 50
    rsi_window: int = 20

    # Divergence thresholds (for LOGGING only)
    divergence_H_threshold: float = 0.1
    divergence_rho_threshold: float = 0.1
    divergence_streak_threshold: int = 20

    # Red-flag thresholds (P3-inherited, LOGGING only)
    cdi_010_observation_enabled: bool = True
    cdi_007_streak_threshold: int = 10
    rsi_collapse_threshold: float = 0.2
    omega_exit_threshold: int = 100
    block_rate_threshold: float = 0.6

    # SHADOW MODE: Must always be True
    shadow_mode: bool = True

    def validate(self) -> List[str]:
        """Validate configuration."""
        raise NotImplementedError("P4 implementation not yet activated")

    def validate_or_raise(self) -> None:
        """Validate and raise on error."""
        raise NotImplementedError("P4 implementation not yet activated")
```

### 6.5 FirstLightResultP4

```python
@dataclass
class FirstLightResultP4:
    """
    Result of P4 shadow experiment.

    Extends P3 result with divergence analysis.
    """

    # Run metadata
    run_id: str = ""
    config_slice: str = ""
    config_runner_type: str = ""
    start_time: str = ""
    end_time: str = ""

    # Cycle summary
    total_cycles_requested: int = 0
    cycles_completed: int = 0

    # P3 metrics (inherited)
    u2_success_rate_final: float = 0.0
    u2_success_rate_trajectory: List[float] = field(default_factory=list)
    delta_p_success: Optional[float] = None

    rfl_abstention_rate_final: float = 0.0
    rfl_abstention_trajectory: List[float] = field(default_factory=list)
    delta_p_abstention: Optional[float] = None

    mean_rsi: float = 0.0
    min_rsi: float = 0.0
    max_rsi: float = 0.0
    rsi_trajectory: List[float] = field(default_factory=list)

    omega_occupancy: float = 0.0
    omega_exit_count: int = 0
    max_omega_exit_streak: int = 0
    omega_occupancy_trajectory: List[float] = field(default_factory=list)

    hard_ok_rate: float = 0.0
    hard_fail_count: int = 0
    max_hard_fail_streak: int = 0
    hard_ok_trajectory: List[float] = field(default_factory=list)

    # Red-flag summary
    total_red_flags: int = 0
    red_flags_by_type: Dict[str, int] = field(default_factory=dict)
    red_flags_by_severity: Dict[str, int] = field(default_factory=dict)

    hypothetical_abort_cycle: Optional[int] = None
    hypothetical_abort_reason: Optional[str] = None

    # P4-specific: Divergence analysis
    total_divergences: int = 0
    divergences_by_type: Dict[str, int] = field(default_factory=dict)
    divergences_by_severity: Dict[str, int] = field(default_factory=dict)
    max_divergence_streak: int = 0
    divergence_rate: float = 0.0

    # Twin accuracy
    twin_success_prediction_accuracy: float = 0.0
    twin_blocked_prediction_accuracy: float = 0.0
    twin_omega_prediction_accuracy: float = 0.0

    # P4-specific: Output paths
    real_cycles_log_path: str = ""
    twin_cycles_log_path: str = ""
    divergence_log_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        raise NotImplementedError("P4 implementation not yet activated")
```

---

## 7. Schema Specifications

### 7.1 Real Cycles Log (`real_cycles.jsonl`)

```json
{
  "schema": "first-light-p4-real-cycle/1.0.0",
  "source": "REAL_RUNNER",
  "cycle": 42,
  "timestamp": "2025-12-09T12:00:42.000000+00:00",
  "mode": "SHADOW",

  "runner": {
    "type": "u2",
    "slice": "arithmetic_simple",
    "success": true,
    "depth": 4,
    "proof_hash": "abc123..."
  },

  "usla_state": {
    "H": 0.75,
    "rho": 0.85,
    "tau": 0.21,
    "beta": 0.05,
    "in_omega": true
  },

  "governance": {
    "real_blocked": false,
    "governance_aligned": true
  },

  "hard_mode": {
    "hard_ok": true
  },

  "abstention": {
    "abstained": false,
    "reason": null
  }
}
```

### 7.2 Twin Cycles Log (`twin_cycles.jsonl`)

```json
{
  "schema": "first-light-p4-twin-cycle/1.0.0",
  "source": "SHADOW_TWIN",
  "real_cycle": 42,
  "timestamp": "2025-12-09T12:00:42.000000+00:00",
  "mode": "SHADOW",

  "predictions": {
    "success": true,
    "blocked": false,
    "in_omega": true,
    "hard_ok": true
  },

  "twin_state": {
    "H": 0.74,
    "rho": 0.84,
    "tau": 0.21,
    "beta": 0.06
  },

  "confidence": {
    "prediction_confidence": 0.92
  }
}
```

### 7.3 Divergence Log (`divergence.jsonl`)

```json
{
  "schema": "first-light-p4-divergence/1.0.0",
  "cycle": 42,
  "timestamp": "2025-12-09T12:00:42.000000+00:00",
  "mode": "SHADOW",

  "divergence_flags": {
    "success_diverged": false,
    "blocked_diverged": false,
    "omega_diverged": false,
    "hard_ok_diverged": false
  },

  "magnitude": {
    "H_delta": 0.01,
    "rho_delta": 0.01,
    "tau_delta": 0.00,
    "beta_delta": -0.01
  },

  "classification": {
    "severity": "NONE",
    "type": "NONE",
    "consecutive_count": 0
  },

  "action": "LOGGED_ONLY"
}
```

### 7.4 P4 Summary Schema Extension

The P4 summary extends the P3 summary with:

```json
{
  "schema": "first-light-p4-summary/1.0.0",

  "divergence_analysis": {
    "total_divergences": 15,
    "by_type": {
      "STATE": 10,
      "OUTCOME": 3,
      "BOTH": 2
    },
    "by_severity": {
      "MINOR": 12,
      "MODERATE": 2,
      "SEVERE": 1
    },
    "max_streak": 3,
    "divergence_rate": 0.015
  },

  "twin_accuracy": {
    "success_prediction": 0.94,
    "blocked_prediction": 0.98,
    "omega_prediction": 0.96
  },

  "output_paths": {
    "real_cycles": "results/first_light_p4/run_id/real_cycles.jsonl",
    "twin_cycles": "results/first_light_p4/run_id/twin_cycles.jsonl",
    "divergence": "results/first_light_p4/run_id/divergence.jsonl"
  }
}
```

---

## 8. Experiment Flow

### 8.1 P4 Execution Sequence

```
1. Initialize FirstLightShadowRunnerP4 with config
   └── Validate shadow_mode=True (INV-01)
   └── Validate adapter is read-only (INV-02)
   └── Initialize P3 components (RedFlagObserver, DeltaPComputer, MetricsAccumulator)
   └── Initialize P4 components (TwinRunner, DivergenceAnalyzer)

2. For each cycle:
   a. Adapter reads telemetry snapshot (READ-ONLY)
   b. Create RealCycleObservation from snapshot
   c. TwinRunner generates TwinCycleObservation (prediction only)
   d. DivergenceAnalyzer compares real vs twin
   e. RedFlagObserver checks for anomalies (LOGGING only)
   f. DeltaPComputer updates metrics (read-only)
   g. MetricsAccumulator updates windows (read-only)
   h. Write JSONL entries

3. Finalize:
   a. Compute final metrics
   b. Generate divergence summary
   c. Write summary.json
   d. Return FirstLightResultP4
```

### 8.2 Shadow Mode Verification Points

At each step, verify:

| Step | Verification | Method |
|------|--------------|--------|
| 1 | Adapter is read-only | Type check, no write methods |
| 2a | Snapshot is immutable | frozen dataclass |
| 2b | No external modification | Pure function |
| 2c | Twin uses only observed data | No external calls |
| 2d | Divergence is logged only | action="LOGGED_ONLY" |
| 2e | Red-flags not enforced | No return value influences control |
| 3 | Results are observation only | No callbacks to runners |

---

## 9. Test Plan

### 9.1 Test Categories

| Category | Count | Purpose |
|----------|-------|---------|
| Import Tests | 10+ | Verify modules import without error |
| Stub Tests | 15+ | Verify NotImplementedError raised |
| Interface Tests | 10+ | Verify interface contracts |
| Schema Tests | 5+ | Verify JSONL schema compliance |
| Invariant Tests | 7+ | Verify absolute invariants |
| Integration Smoke | 2+ | Verify component wiring |

### 9.2 Stub Test Examples

```python
class TestP4DesignStubs:
    """P4 stub tests - all implementations must raise NotImplementedError."""

    def test_config_p4_validate_raises(self) -> None:
        """Verify FirstLightConfigP4.validate() is stub."""
        from backend.topology.first_light.config_p4 import FirstLightConfigP4
        config = FirstLightConfigP4()
        with pytest.raises(NotImplementedError, match="P4 implementation not yet activated"):
            config.validate()

    def test_runner_p4_run_raises(self) -> None:
        """Verify FirstLightShadowRunnerP4.run() is stub."""
        from backend.topology.first_light.runner_p4 import FirstLightShadowRunnerP4
        from backend.topology.first_light.config_p4 import FirstLightConfigP4
        config = FirstLightConfigP4()
        with pytest.raises(NotImplementedError, match="P4 implementation not yet activated"):
            FirstLightShadowRunnerP4(config)

    def test_adapter_get_snapshot_raises(self) -> None:
        """Verify USLAIntegrationAdapter.get_current_snapshot() is stub."""
        from backend.topology.first_light.telemetry_adapter import USLAIntegrationAdapter
        with pytest.raises(NotImplementedError, match="P4 implementation not yet activated"):
            USLAIntegrationAdapter(None, "u2")
```

### 9.3 Invariant Test Examples

```python
class TestP4Invariants:
    """Tests verifying P4 absolute invariants."""

    def test_invariant_01_shadow_mode_required(self) -> None:
        """INV-01: Shadow mode must be True."""
        from backend.topology.first_light.config_p4 import FirstLightConfigP4
        config = FirstLightConfigP4(shadow_mode=True)
        # When implementation exists, shadow_mode=False should raise

    def test_invariant_06_mode_declaration(self) -> None:
        """INV-06: All structures include mode='SHADOW'."""
        from backend.topology.first_light.data_structures_p4 import RealCycleObservation
        # Verify schema compliance when implemented
```

---

## 10. Non-Goals

The following are explicitly **NOT** goals for P4:

| Non-Goal | Reason | Future Phase |
|----------|--------|--------------|
| Active governance modification | Violates SHADOW MODE | P5+ |
| Abort enforcement | Violates SHADOW MODE | P5+ |
| Real-time alerting | Beyond observation scope | TBD |
| Runner feedback | Violates observer effect constraint | Never in shadow |
| Performance optimization | Focus is correctness | Post-implementation |
| Distributed execution | Single-process observation only | TBD |

---

## 11. Future Bridge to P5

### 11.1 P5 Scope (NOT AUTHORIZED)

P5 would transition from observation to limited intervention. This requires:

1. **Governance Authorization** — Explicit approval for active mode
2. **Safety Verification** — Formal proof of bounded intervention
3. **Rollback Capability** — Ability to revert to shadow mode
4. **Audit Trail** — Complete history of interventions

### 11.2 P4 → P5 Interface Requirements

P4 must establish:

- Clean separation between observation and intervention code paths
- Metrics that would inform intervention decisions (but don't act)
- Schema extensions for intervention logging
- Test infrastructure for intervention verification

**P5 is NOT authorized. P4 is observation-only.**

---

## 12. Implementation Skeletons

### 12.1 File Structure

```
backend/topology/first_light/
├── __init__.py                  # Updated with P4 exports
├── config_p4.py                 # FirstLightConfigP4 (STUB)
├── runner_p4.py                 # FirstLightShadowRunnerP4 (STUB)
├── telemetry_adapter.py         # USLAIntegrationAdapter (STUB)
├── divergence_analyzer.py       # DivergenceAnalyzer (STUB)
├── data_structures_p4.py        # P4 data structures (STUB)
└── schemas_p4.py                # P4 schema definitions (STUB)

tests/first_light/
├── __init__.py
├── test_first_light_stubs.py    # P3 import tests
├── test_first_light_behavior.py # P3 behavior tests
└── test_p4_design_stubs.py      # P4 stub tests (NEW)
```

### 12.2 Skeleton Requirements

All P4 skeleton files MUST:

1. Include SHADOW MODE docstring at module level
2. Define all classes and functions with proper signatures
3. Raise `NotImplementedError("P4 implementation not yet activated")` in all method bodies
4. Include type hints
5. Export via `__all__`

---

## 13. Authorization Gates

### 13.1 Current Authorization (P4 Design Freeze)

| Capability | Authorized | Notes |
|------------|------------|-------|
| Design specification | YES | This document |
| Code skeletons/stubs | YES | NotImplementedError only |
| Import smoke tests | YES | Minimal validation |
| Interface definitions | YES | Abstract methods |
| Schema documentation | YES | JSONL formats |

### 13.2 NOT Authorized (Requires Future Phase)

| Capability | Phase Required |
|------------|----------------|
| Working FirstLightShadowRunnerP4 | Explicit P4 execution auth |
| Real telemetry coupling | Explicit P4 execution auth |
| Divergence analysis execution | Explicit P4 execution auth |
| P5 intervention features | Phase X P5 (not designed) |

---

## Implementation Authorized

> **DESIGN DOCUMENT STATUS: APPROVED FOR SKELETON CREATION**
>
> This Phase X P4 Specification is approved for:
> - Documentation in `docs/system_law/`
> - Skeleton module creation with `NotImplementedError` stubs
> - Stub test creation for import and interface validation
>
> Implementation of actual P4 logic requires separate authorization.

---

*Document Version: 1.0.0*
*Last Updated: 2025-12-09*
*Status: Design Freeze (Stubs Only)*
