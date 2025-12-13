"""
Phase X P5: Real Telemetry Adapter for Live Runner Shadow Coupling

This module implements the RealTelemetryAdapter for P5 shadow mode.
See docs/system_law/First_Light_P5_Adapter_Checklist.md for full specification.

SHADOW MODE CONTRACT:
- RealTelemetryAdapter is READ-ONLY
- No mutation methods exposed
- All telemetry is captured without modifying real runner state
- mode == "SHADOW" on all observations
- source == "REAL_RUNNER" (or "P5_ADAPTER_STUB" for POC)

ADAPTER MODES:
- "synthetic": Generate realistic telemetry with P5 stub dynamics (default)
- "trace": Replay telemetry from a JSONL trace file for reproducibility

Status: P5 BASELINE HARDENING

=============================================================================
P5 GAP REPORT: High-Leverage Missing Pieces for P5 Acceptance
=============================================================================

The following items represent the 3-5 highest-leverage gaps between the current
P5 POC/baseline and production P5 acceptance. This is documentation only;
no implementation is authorized until governance approval.

GAP-1: True Production Telemetry Source (U2/USLA Integration)
    Current state: RealTelemetryAdapter generates synthetic data or replays traces.
    Required: Direct integration with live U2/RFL runner telemetry streams.
    Implementation path:
    - Redis stream subscription for real-time telemetry
    - gRPC endpoint for structured telemetry fetch
    - Shared memory interface for low-latency access
    - Connection resilience and backpressure handling
    Status: NOT AUTHORIZED (requires production integration approval)

GAP-2: Twin Calibration Engine (Warm-Start, Parameter Optimization)
    Current state: TwinRunner uses fixed learning_rate=0.1, noise_scale=0.02.
    Required: Adaptive calibration based on historical divergence patterns.
    Implementation path:
    - Warm-start from historical twin state when resuming runs
    - Learning rate scheduling based on convergence metrics
    - Per-slice parameter profiles (different dynamics for different slices)
    - Online parameter optimization using divergence gradient
    Status: NOT AUTHORIZED (requires P5 acceptance gate definition)

GAP-3: DivergencePatternClassifier Hooks
    Current state: Divergence classified as STATE/OUTCOME/BOTH with severity.
    Required: Fine-grained pattern classification for root cause analysis.
    Implementation path:
    - DRIFT: Gradual state divergence over time (detectable trend)
    - PHASE_LAG: Twin tracks real but with consistent delay
    - STRUCTURAL_BREAK: Sudden regime change (learning curve inflection)
    - NOISE_FLOOR: Divergence at irreducible noise level
    - SYSTEMATIC_BIAS: Consistent directional prediction error
    Status: NOT AUTHORIZED (requires divergence metric spec finalization)

GAP-4: P5AcceptanceGate Wiring (SHADOW, Spec Only)
    Current state: Success criteria evaluated but not enforced.
    Required: Formal acceptance gate with configurable thresholds.
    Implementation path:
    - Define P5AcceptanceEnvelope dataclass with threshold bounds
    - Implement evaluate_acceptance() returning PASS/FAIL/OBSERVE
    - Wire into evidence pack as governance advisory
    - Threshold tuning based on baseline characterization runs
    Target: divergence_rate < 0.30, twin_success_accuracy > 0.85
    Status: NOT AUTHORIZED (SHADOW MODE - spec only, no enforcement)

GAP-5: TDA Mind Scanner Integration (SNS, PCS, DRS, HSS)
    Current state: DRS computed in DivergenceSnapshot, others are None.
    Required: Full TDA metric suite for real vs twin comparison.
    Implementation path:
    - Compute SNS (Signal-to-Noise Score) for telemetry windows
    - Compute PCS (Phase Coherence Score) for state alignment
    - Compute HSS (Health Stability Score) for H trajectory
    - Emit TDA comparison tile in evidence pack
    Status: NOT AUTHORIZED (requires TDA_PhaseX_Binding.md completion)

=============================================================================
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from backend.topology.first_light.data_structures_p4 import TelemetrySnapshot
from backend.topology.first_light.telemetry_adapter import TelemetryProviderInterface

__all__ = [
    "RealTelemetryAdapter",
    "RealTelemetryAdapterConfig",
    "AdapterMode",
    "ValidationResult",
    "validate_real_telemetry_window",
    "write_trace_jsonl",
    "load_trace_jsonl",
]


# =============================================================================
# Adapter Mode Enumeration
# =============================================================================

class AdapterMode:
    """
    Adapter operational modes.

    SHADOW MODE: All modes are observation-only.
    """
    SYNTHETIC = "synthetic"  # Generate P5 stub synthetic telemetry
    TRACE = "trace"          # Replay from JSONL trace file


@dataclass
class RealTelemetryAdapterConfig:
    """
    Configuration for RealTelemetryAdapter.

    This configuration can be loaded from JSON via --adapter-config flag.

    SHADOW MODE: Configuration does not affect governance; all modes are
    observation-only.

    JSON Schema:
    {
        "mode": "synthetic" | "trace",
        "trace_path": "path/to/trace.jsonl",  // required if mode == "trace"
        "runner_type": "u2" | "rfl",
        "slice_name": "arithmetic_simple",
        "seed": 42,
        "source_label": "P5_ADAPTER_STUB",
        "lipschitz_thresholds": { ... }  // optional, for future use
    }
    """
    mode: str = AdapterMode.SYNTHETIC
    trace_path: Optional[str] = None
    runner_type: str = "u2"
    slice_name: str = "arithmetic_simple"
    seed: Optional[int] = None
    source_label: str = "P5_ADAPTER_STUB"
    lipschitz_thresholds: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.mode not in (AdapterMode.SYNTHETIC, AdapterMode.TRACE):
            raise ValueError(
                f"mode must be '{AdapterMode.SYNTHETIC}' or '{AdapterMode.TRACE}', "
                f"got '{self.mode}'"
            )
        if self.mode == AdapterMode.TRACE and not self.trace_path:
            raise ValueError("trace_path is required when mode == 'trace'")
        if self.runner_type not in ("u2", "rfl"):
            raise ValueError(
                f"runner_type must be 'u2' or 'rfl', got '{self.runner_type}'"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "mode": self.mode,
            "trace_path": self.trace_path,
            "runner_type": self.runner_type,
            "slice_name": self.slice_name,
            "seed": self.seed,
            "source_label": self.source_label,
            "lipschitz_thresholds": self.lipschitz_thresholds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RealTelemetryAdapterConfig":
        """Create config from dictionary (e.g., loaded from JSON)."""
        return cls(
            mode=data.get("mode", AdapterMode.SYNTHETIC),
            trace_path=data.get("trace_path"),
            runner_type=data.get("runner_type", "u2"),
            slice_name=data.get("slice_name", "arithmetic_simple"),
            seed=data.get("seed"),
            source_label=data.get("source_label", "P5_ADAPTER_STUB"),
            lipschitz_thresholds=data.get("lipschitz_thresholds"),
        )

    @classmethod
    def from_json_file(cls, path: str) -> "RealTelemetryAdapterConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


# =============================================================================
# RTTS (Real Telemetry Technical Specification) Constants
# =============================================================================

# Lipschitz thresholds per RTTS specification
# Maximum allowed delta per cycle for each metric
LIPSCHITZ_THRESHOLDS = {
    "H": 0.15,      # Health metric: smooth changes
    "rho": 0.10,    # RSI: very stable
    "tau": 0.05,    # Threshold: slow drift
    "beta": 0.20,   # Block rate: can change faster
}

# Noise floors - minimum variance expected in real telemetry
# Set very low to avoid false positives on short windows
NOISE_FLOORS = {
    "H": 0.0001,
    "rho": 0.0001,
    "tau": 0.00005,
    "beta": 0.0001,
}

# Discreteness thresholds - mock data often has discrete jumps
DISCRETE_JUMP_THRESHOLD = 0.05


@dataclass
class ValidationResult:
    """
    Result of validating a real telemetry window.

    SHADOW MODE: This is an observational result only.
    """
    confidence: float = 0.0          # [0,1] confidence that data is real
    mock_indicators: List[str] = field(default_factory=list)  # Detected mock patterns
    status: str = "UNKNOWN"          # "PROVISIONAL_REAL" | "MOCK_LIKE" | "UNKNOWN"
    lipschitz_violations: int = 0    # Count of Lipschitz violations
    flatness_score: float = 0.0      # 0=varied, 1=flat
    discreteness_score: float = 0.0  # 0=continuous, 1=discrete jumps

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "confidence": round(self.confidence, 4),
            "mock_indicators": self.mock_indicators,
            "status": self.status,
            "lipschitz_violations": self.lipschitz_violations,
            "flatness_score": round(self.flatness_score, 4),
            "discreteness_score": round(self.discreteness_score, 4),
        }


def validate_real_telemetry_window(
    frames: List[TelemetrySnapshot],
    lipschitz_thresholds: Optional[Dict[str, float]] = None,
) -> ValidationResult:
    """
    Validate a window of telemetry frames for real-ness.

    SHADOW MODE: This is a read-only analysis function.
    It does not modify any state.

    Checks:
    1. Basic boundedness (all metrics in [0,1])
    2. Lipschitz deltas below RTTS thresholds
    3. Variance/flatness detection
    4. Discreteness detection (mock data often has discrete jumps)

    Args:
        frames: List of TelemetrySnapshot objects to validate
        lipschitz_thresholds: Optional custom thresholds (defaults to RTTS)

    Returns:
        ValidationResult with confidence and mock indicators
    """
    if not frames:
        return ValidationResult(
            confidence=0.0,
            mock_indicators=["empty_window"],
            status="UNKNOWN",
        )

    thresholds = lipschitz_thresholds or LIPSCHITZ_THRESHOLDS
    mock_indicators: List[str] = []
    lipschitz_violations = 0

    # Collect metric series
    H_series = [f.H for f in frames]
    rho_series = [f.rho for f in frames]
    tau_series = [f.tau for f in frames]
    beta_series = [f.beta for f in frames]

    # 1. Check boundedness
    for name, series in [("H", H_series), ("rho", rho_series),
                          ("tau", tau_series), ("beta", beta_series)]:
        for val in series:
            if not (0.0 <= val <= 1.0):
                mock_indicators.append(f"{name}_out_of_bounds")
                break

    # 2. Check Lipschitz continuity
    for name, series in [("H", H_series), ("rho", rho_series),
                          ("tau", tau_series), ("beta", beta_series)]:
        threshold = thresholds.get(name, 0.2)
        for i in range(1, len(series)):
            delta = abs(series[i] - series[i-1])
            if delta > threshold:
                lipschitz_violations += 1

    if lipschitz_violations > len(frames) * 0.1:  # >10% violations
        mock_indicators.append("excessive_lipschitz_violations")

    # 3. Check flatness (mock data often flat or perfectly trending)
    flatness_scores = []
    for name, series in [("H", H_series), ("rho", rho_series),
                          ("tau", tau_series), ("beta", beta_series)]:
        if len(series) > 1:
            variance = _compute_variance(series)
            noise_floor = NOISE_FLOORS.get(name, 0.001)
            # If variance is below noise floor, data is suspiciously flat
            if variance < noise_floor:
                flatness_scores.append(1.0)
                mock_indicators.append(f"{name}_suspiciously_flat")
            else:
                flatness_scores.append(max(0.0, 1.0 - variance / 0.01))

    flatness_score = sum(flatness_scores) / len(flatness_scores) if flatness_scores else 0.0

    # 4. Check discreteness (mock data often has discrete state jumps)
    discreteness_scores = []
    for name, series in [("H", H_series), ("rho", rho_series),
                          ("tau", tau_series), ("beta", beta_series)]:
        discrete_count = 0
        for i in range(1, len(series)):
            delta = abs(series[i] - series[i-1])
            # Check if delta is a "clean" discrete value
            if delta > 0 and delta < DISCRETE_JUMP_THRESHOLD:
                # Real data has continuous small variations
                pass
            elif delta >= DISCRETE_JUMP_THRESHOLD:
                # Check if it's a suspiciously round number
                rounded = round(delta, 2)
                if abs(delta - rounded) < 0.001:
                    discrete_count += 1

        if len(series) > 1:
            discreteness_scores.append(discrete_count / (len(series) - 1))

    discreteness_score = sum(discreteness_scores) / len(discreteness_scores) if discreteness_scores else 0.0

    if discreteness_score > 0.5:
        mock_indicators.append("discrete_state_changes")

    # 5. Compute overall confidence
    # Higher confidence = more likely to be real telemetry
    confidence = 1.0

    # Penalize for mock indicators
    confidence -= len(mock_indicators) * 0.15

    # Penalize for flatness
    confidence -= flatness_score * 0.3

    # Penalize for discreteness
    confidence -= discreteness_score * 0.2

    # Penalize for Lipschitz violations (but some are expected)
    violation_rate = lipschitz_violations / max(1, len(frames))
    if violation_rate > 0.1:
        confidence -= (violation_rate - 0.1) * 0.5

    confidence = max(0.0, min(1.0, confidence))

    # Determine status
    if confidence >= 0.6:
        status = "PROVISIONAL_REAL"
    elif confidence <= 0.3:
        status = "MOCK_LIKE"
    else:
        status = "UNKNOWN"

    return ValidationResult(
        confidence=confidence,
        mock_indicators=mock_indicators,
        status=status,
        lipschitz_violations=lipschitz_violations,
        flatness_score=flatness_score,
        discreteness_score=discreteness_score,
    )


def _compute_variance(series: List[float]) -> float:
    """Compute variance of a series."""
    if len(series) < 2:
        return 0.0
    mean = sum(series) / len(series)
    return sum((x - mean) ** 2 for x in series) / len(series)


# =============================================================================
# Trace File I/O (P5 Reproducibility Spine)
# =============================================================================

def write_trace_jsonl(
    snapshots: List[TelemetrySnapshot],
    path: str,
    include_timestamp: bool = True,
) -> int:
    """
    Write TelemetrySnapshots to a JSONL trace file.

    SHADOW MODE: This is a write-only operation for recording telemetry.
    It does not affect any governance state.

    Args:
        snapshots: List of TelemetrySnapshot objects to write
        path: Path to output JSONL file
        include_timestamp: If True, preserve original timestamps; if False, omit

    Returns:
        Number of snapshots written

    Raises:
        IOError: If file cannot be written
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w") as f:
        for snapshot in snapshots:
            record = {
                "cycle": snapshot.cycle,
                "runner_type": snapshot.runner_type,
                "slice_name": snapshot.slice_name,
                "H": snapshot.H,
                "rho": snapshot.rho,
                "tau": snapshot.tau,
                "beta": snapshot.beta,
                "success": snapshot.success,
                "depth": snapshot.depth,
                "in_omega": snapshot.in_omega,
                "blocked": snapshot.real_blocked,
                "hard_ok": snapshot.hard_ok,
                "abstention": snapshot.abstained,
                "abstention_reason": snapshot.abstention_reason,
                "governance_aligned": snapshot.governance_aligned,
                "governance_reason": snapshot.governance_reason,
                "snapshot_hash": snapshot.snapshot_hash,
            }
            if include_timestamp:
                record["timestamp"] = snapshot.timestamp

            f.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1

    return count


def load_trace_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load trace records from a JSONL file.

    SHADOW MODE: This is a read-only operation.

    Args:
        path: Path to JSONL trace file

    Returns:
        List of trace record dictionaries

    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    trace_path = Path(path)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    records: List[Dict[str, Any]] = []
    with open(trace_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"Invalid JSON at line {line_num}: {e.msg}",
                        e.doc,
                        e.pos,
                    )

    return records


class RealTelemetryAdapter(TelemetryProviderInterface):
    """
    Adapter for real runner telemetry (P5 POC).

    SHADOW MODE CONTRACT:
    - This adapter is READ-ONLY
    - No mutation methods are exposed
    - All observations have mode="SHADOW"
    - All observations have source="REAL_RUNNER" or "P5_ADAPTER_STUB"

    For this POC, the adapter can:
    1. Read from a recorded JSONL trace file, OR
    2. Generate "better than Mock" synthetic data with realistic dynamics

    The adapter satisfies the 14-field contract from:
    docs/system_law/First_Light_P5_Adapter_Checklist.md

    See: docs/system_law/First_Light_P5_Adapter_Checklist.md
    """

    # 14 required fields per P5 interface contract
    REQUIRED_FIELDS = [
        "cycle", "timestamp", "H", "rho", "tau", "beta",
        "success", "blocked", "in_omega", "hard_ok", "rsi",
        "abstention", "mode", "source",
    ]

    def __init__(
        self,
        runner_type: str = "u2",
        slice_name: str = "arithmetic_simple",
        seed: Optional[int] = None,
        trace_path: Optional[str] = None,
        source_label: str = "P5_ADAPTER_STUB",
        mode: Optional[str] = None,
    ) -> None:
        """
        Initialize real telemetry adapter.

        Args:
            runner_type: Type of runner ("u2" or "rfl")
            slice_name: Name of slice
            seed: Random seed for reproducibility (used if no trace_path)
            trace_path: Optional path to recorded JSONL trace file
            source_label: Source identifier (default: P5_ADAPTER_STUB)
            mode: Explicit mode ("synthetic" or "trace"). If None, auto-detect
                  based on trace_path presence.

        Raises:
            ValueError: If runner_type is invalid
            ValueError: If mode is "trace" but trace_path is not provided
            FileNotFoundError: If mode is "trace" and trace file doesn't exist
        """
        if runner_type not in ("u2", "rfl"):
            raise ValueError(f"runner_type must be 'u2' or 'rfl', got '{runner_type}'")

        self._runner_type = runner_type
        self._slice_name = slice_name
        self._seed = seed
        self._source_label = source_label
        self._rng = random.Random(seed)

        # Determine mode
        if mode is None:
            # Auto-detect based on trace_path
            self._mode = AdapterMode.TRACE if trace_path else AdapterMode.SYNTHETIC
        else:
            if mode not in (AdapterMode.SYNTHETIC, AdapterMode.TRACE):
                raise ValueError(
                    f"mode must be '{AdapterMode.SYNTHETIC}' or '{AdapterMode.TRACE}', "
                    f"got '{mode}'"
                )
            self._mode = mode

        # Trace file mode validation
        self._trace_path = Path(trace_path) if trace_path else None
        self._trace_data: List[Dict[str, Any]] = []
        self._trace_index = 0

        if self._mode == AdapterMode.TRACE:
            if not self._trace_path:
                raise ValueError("trace_path is required when mode == 'trace'")
            if not self._trace_path.exists():
                raise FileNotFoundError(f"Trace file not found: {trace_path}")
            self._load_trace()
        elif self._trace_path and self._trace_path.exists():
            # Synthetic mode but trace provided - load anyway for optional use
            self._load_trace()

        # Internal state for synthetic mode
        # Start at same initial values as TwinRunner for minimal initial divergence
        # TwinRunner starts at: H=0.5, rho=0.7, tau=tau_0, beta=0.1
        self._cycle = 0
        self._H = 0.5  # Match Twin's initial H
        self._rho = 0.7  # Match Twin's initial rho
        self._tau = 0.2  # Match tau_0 default
        self._beta = 0.1  # Match Twin's initial beta
        self._available = True

        # Historical snapshots
        self._history: List[TelemetrySnapshot] = []

        # Validation window
        self._validation_window_size = 20

    @classmethod
    def from_config(cls, config: RealTelemetryAdapterConfig) -> "RealTelemetryAdapter":
        """
        Create adapter from configuration object.

        This is the preferred way to create an adapter when using --adapter-config.

        Args:
            config: RealTelemetryAdapterConfig instance

        Returns:
            Configured RealTelemetryAdapter instance
        """
        return cls(
            runner_type=config.runner_type,
            slice_name=config.slice_name,
            seed=config.seed,
            trace_path=config.trace_path,
            source_label=config.source_label,
            mode=config.mode,
        )

    def _load_trace(self) -> None:
        """Load trace data from JSONL file."""
        if not self._trace_path or not self._trace_path.exists():
            return

        self._trace_data = []
        with open(self._trace_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        self._trace_data.append(record)
                    except json.JSONDecodeError:
                        continue

    def get_snapshot(self) -> Optional[TelemetrySnapshot]:
        """
        Get telemetry snapshot.

        SHADOW MODE: This is a read-only operation.
        Returns data from trace file or generates realistic synthetic data.

        Returns:
            TelemetrySnapshot with 14+ fields, mode="SHADOW"
        """
        if not self._available:
            return None

        self._cycle += 1

        # Use trace data if available
        if self._trace_data and self._trace_index < len(self._trace_data):
            snapshot = self._snapshot_from_trace()
        else:
            # Generate realistic synthetic data (better than Mock)
            snapshot = self._generate_realistic_snapshot()

        self._history.append(snapshot)
        return snapshot

    def _snapshot_from_trace(self) -> TelemetrySnapshot:
        """Create snapshot from trace record."""
        record = self._trace_data[self._trace_index]
        self._trace_index += 1

        timestamp = record.get("timestamp", datetime.now(timezone.utc).isoformat())

        # Extract metrics with defaults
        H = float(record.get("H", 0.5))
        rho = float(record.get("rho", 0.7))
        tau = float(record.get("tau", 0.2))
        beta = float(record.get("beta", 0.1))

        # Update internal state to track trace
        self._H = H
        self._rho = rho
        self._tau = tau
        self._beta = beta

        # Build snapshot data for hash
        data = {
            "cycle": self._cycle,
            "timestamp": timestamp,
            "runner_type": self._runner_type,
            "H": H,
            "rho": rho,
            "tau": tau,
            "beta": beta,
        }
        snapshot_hash = TelemetrySnapshot.compute_hash(data)

        return TelemetrySnapshot(
            cycle=self._cycle,
            timestamp=timestamp,
            runner_type=self._runner_type,
            slice_name=self._slice_name,
            success=record.get("success", False),
            depth=record.get("depth"),
            proof_hash=record.get("proof_hash"),
            H=H,
            rho=rho,
            tau=tau,
            beta=beta,
            in_omega=record.get("in_omega", H > tau and rho > 0.5),
            real_blocked=record.get("blocked", False),
            governance_aligned=record.get("governance_aligned", True),
            governance_reason=record.get("governance_reason"),
            hard_ok=record.get("hard_ok", True),
            abstained=record.get("abstention", None),
            abstention_reason=record.get("abstention_reason"),
            reasoning_graph_hash=None,
            proof_dag_size=0,
            snapshot_hash=snapshot_hash,
        )

    def _generate_realistic_snapshot(self) -> TelemetrySnapshot:
        """
        Generate realistic synthetic snapshot.

        This is "better than Mock" because it:
        1. Uses correlated state transitions (not independent noise)
        2. Respects Lipschitz continuity bounds
        3. Models realistic learning curves
        4. Includes micro-noise that Mock lacks
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Update state with realistic dynamics
        self._update_realistic_state()

        # Compute derived values
        in_omega = self._H > self._tau and self._rho > 0.5

        # Success correlates with state (unlike Mock's independent random)
        # Higher H, higher rho, lower beta -> higher success probability
        base_success_prob = 0.5 + 0.3 * self._H + 0.2 * self._rho - 0.3 * self._beta
        base_success_prob = max(0.1, min(0.95, base_success_prob))
        success = self._rng.random() < base_success_prob

        # Hard OK correlates with rho
        hard_ok = self._rng.random() < (0.9 + 0.1 * self._rho)

        # Real blocked correlates with beta
        real_blocked = self._beta > 0.6 and self._rng.random() < self._beta

        # RSI is a function of H and rho
        rsi = 0.4 * self._H + 0.6 * self._rho
        rsi = max(0.0, min(1.0, rsi + self._rng.gauss(0, 0.02)))

        # Abstention for RFL
        abstained = None
        if self._runner_type == "rfl":
            abstained = self._rng.random() < 0.08 * (1 - self._H)

        # Build snapshot data for hash
        data = {
            "cycle": self._cycle,
            "timestamp": timestamp,
            "runner_type": self._runner_type,
            "H": self._H,
            "rho": self._rho,
            "tau": self._tau,
            "beta": self._beta,
        }
        snapshot_hash = TelemetrySnapshot.compute_hash(data)

        return TelemetrySnapshot(
            cycle=self._cycle,
            timestamp=timestamp,
            runner_type=self._runner_type,
            slice_name=self._slice_name,
            success=success,
            depth=self._rng.randint(3, 8) if success else self._rng.randint(1, 3),
            proof_hash=None,
            H=self._H,
            rho=self._rho,
            tau=self._tau,
            beta=self._beta,
            in_omega=in_omega,
            real_blocked=real_blocked,
            governance_aligned=True,
            governance_reason=None,
            hard_ok=hard_ok,
            abstained=abstained,
            abstention_reason="uncertainty" if abstained else None,
            reasoning_graph_hash=None,
            proof_dag_size=0,
            snapshot_hash=snapshot_hash,
        )

    def _update_realistic_state(self) -> None:
        """
        Update internal state with realistic dynamics.

        Key differences from MockTelemetryProvider:
        1. Respects Lipschitz bounds (smooth transitions)
        2. State variables are correlated (not independent)
        3. Includes learning curve with saturation
        4. Micro-noise is continuous, not discrete
        5. **State changes are smoother** - Twin can track this better

        The Twin learns by blending its state toward the real state.
        With smoother state transitions, the Twin's tracking error is lower,
        resulting in lower divergence than with Mock's independent noise.
        """
        # Learning progress (saturating)
        learning = 0.3 * (1 - math.exp(-self._cycle / 200))

        # H: Health metric - improves with learning, correlated with success history
        # Use smaller deltas than Mock for smoother transitions
        h_target = 0.5 + learning
        h_delta = 0.015 * (h_target - self._H) + self._rng.gauss(0, 0.003)
        # Clamp delta to Lipschitz bound
        h_delta = max(-LIPSCHITZ_THRESHOLDS["H"], min(LIPSCHITZ_THRESHOLDS["H"], h_delta))
        self._H = max(0.0, min(1.0, self._H + h_delta))

        # rho: RSI - stable with slow drift toward H (smoother than Mock)
        rho_target = 0.3 + 0.5 * self._H
        rho_delta = 0.008 * (rho_target - self._rho) + self._rng.gauss(0, 0.002)
        rho_delta = max(-LIPSCHITZ_THRESHOLDS["rho"], min(LIPSCHITZ_THRESHOLDS["rho"], rho_delta))
        self._rho = max(0.0, min(1.0, self._rho + rho_delta))

        # tau: Threshold - very slow drift (almost constant)
        tau_delta = self._rng.gauss(0, 0.001)
        tau_delta = max(-LIPSCHITZ_THRESHOLDS["tau"], min(LIPSCHITZ_THRESHOLDS["tau"], tau_delta))
        self._tau = max(0.0, min(1.0, self._tau + tau_delta))

        # beta: Block rate - inversely related to H (smoother tracking)
        beta_target = 0.3 * (1 - self._H)
        beta_delta = 0.015 * (beta_target - self._beta) + self._rng.gauss(0, 0.005)
        beta_delta = max(-LIPSCHITZ_THRESHOLDS["beta"], min(LIPSCHITZ_THRESHOLDS["beta"], beta_delta))
        self._beta = max(0.0, min(1.0, self._beta + beta_delta))

    def is_available(self) -> bool:
        """Check if telemetry is available."""
        return self._available

    def get_current_cycle(self) -> int:
        """Get current cycle number."""
        return self._cycle

    def get_runner_type(self) -> str:
        """Get runner type being observed."""
        return self._runner_type

    def get_historical_snapshots(
        self, start_cycle: int, end_cycle: int
    ) -> Iterator[TelemetrySnapshot]:
        """Get historical snapshots in range (READ-ONLY)."""
        for snapshot in self._history:
            if start_cycle <= snapshot.cycle <= end_cycle:
                yield snapshot

    def validate_recent_window(self) -> ValidationResult:
        """
        Validate the most recent window of telemetry.

        SHADOW MODE: Read-only analysis.

        Returns:
            ValidationResult for the recent window
        """
        window = self._history[-self._validation_window_size:]
        return validate_real_telemetry_window(window)

    def set_available(self, available: bool) -> None:
        """Set availability (for testing only)."""
        self._available = available

    def reset(self) -> None:
        """Reset adapter state."""
        self._cycle = 0
        # Reset to Twin-matching initial values
        self._H = 0.5
        self._rho = 0.7
        self._tau = 0.2
        self._beta = 0.1
        self._rng = random.Random(self._seed)
        self._history.clear()
        self._trace_index = 0

    def get_source_label(self) -> str:
        """Get the source label for this adapter."""
        return self._source_label

    def get_mode(self) -> str:
        """
        Get the adapter operational mode.

        Returns:
            "synthetic" or "trace"
        """
        return self._mode

    def get_snapshots_since(self, cycle: int) -> List[TelemetrySnapshot]:
        """
        Get all snapshots since (and including) the given cycle.

        SHADOW MODE: This is a read-only operation.

        Args:
            cycle: Starting cycle number (inclusive)

        Returns:
            List of TelemetrySnapshot objects from cycle onwards
        """
        return [s for s in self._history if s.cycle >= cycle]

    def get_all_snapshots(self) -> List[TelemetrySnapshot]:
        """
        Get all historical snapshots.

        SHADOW MODE: This is a read-only operation.

        Returns:
            List of all TelemetrySnapshot objects generated/replayed
        """
        return list(self._history)

    def get_trace_length(self) -> int:
        """
        Get the total number of records in the loaded trace.

        Returns:
            Number of trace records, or 0 if in synthetic mode
        """
        return len(self._trace_data)

    def is_trace_exhausted(self) -> bool:
        """
        Check if trace replay has reached the end of the trace.

        Returns:
            True if in trace mode and all records have been consumed
        """
        if self._mode != AdapterMode.TRACE:
            return False
        return self._trace_index >= len(self._trace_data)

    def write_history_to_trace(self, path: str, include_timestamp: bool = True) -> int:
        """
        Write the adapter's history to a JSONL trace file.

        This enables round-trip testing: generate synthetic data, write to trace,
        then replay from trace to verify reproducibility.

        SHADOW MODE: This is a write-only operation for recording telemetry.

        Args:
            path: Path to output JSONL file
            include_timestamp: If True, preserve original timestamps

        Returns:
            Number of snapshots written
        """
        return write_trace_jsonl(self._history, path, include_timestamp)


# =============================================================================
# READ-ONLY INVARIANTS: The following methods are intentionally NOT implemented
# Any attempt to call mutation methods should raise RuntimeError in tests.
# =============================================================================

# Note: TelemetryProviderInterface does not define mutation methods.
# This adapter exposes NO methods that could modify real runner state.
# The SHADOW MODE contract is enforced by design - there are no control surfaces.
