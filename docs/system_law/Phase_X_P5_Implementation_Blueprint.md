# Phase X P5: Implementation Blueprint

**Document ID**: P5-IMPL-BP-2025-001
**Status**: IMPLEMENTATION BLUEPRINT
**Classification**: System Law — Execution Fleet Guidance
**Last Updated**: 2025-12-11
**Upstream Spec**: [Real_Telemetry_Topology_Spec.md](Real_Telemetry_Topology_Spec.md)

---

## Executive Summary

This document translates the P5 Real Telemetry Topology Specification (RTTS) into concrete module designs, class signatures, and test plans suitable for execution by Codex/Manus agents.

**SHADOW MODE INVARIANT**: All P5 code is observational only. No control surfaces, no gating, no aborts. Diagnostics and evidence generation only.

**Execution Model**: Each section below is decomposed into discrete implementation tasks. Tasks marked `[CODEX]` are ready for autonomous implementation. Tasks marked `[ARCHITECT]` require human review before or after implementation.

---

## 1. RealTelemetryAdapter Implementation Plan

### 1.1 Module Location

```
backend/topology/first_light/real_telemetry_adapter.py
```

### 1.2 Core Classes

#### 1.2.1 TelemetryProviderInterface (Reference)

The adapter must implement the existing interface. Locate and conform to:

```python
# Expected interface (verify in backend/topology/first_light/telemetry_provider.py)
class TelemetryProviderInterface(Protocol):
    def get_snapshot(self, cycle: int) -> TelemetrySnapshot: ...
    def reset(self) -> None: ...
```

#### 1.2.2 RealTelemetryAdapter

```python
@dataclass
class RealTelemetryConfig:
    """Configuration for real telemetry ingestion."""
    source_type: Literal["usla_live", "usla_replay", "file"]
    source_path: Optional[str] = None  # For file/replay modes
    validation_window_size: int = 200
    validation_frequency: int = 50
    enable_mock_detection: bool = True
    instrumentation_id: str = "usla-prod-001"


class RealTelemetryAdapter:
    """
    Real telemetry ingestion adapter implementing TelemetryProviderInterface.

    SHADOW MODE CONTRACT:
    - Read-only access to telemetry source
    - No control signals emitted
    - All validation results are advisory only
    - Failures logged but never cause aborts
    """

    def __init__(
        self,
        config: RealTelemetryConfig,
        validator: Optional["TelemetryValidator"] = None,
        mock_detector: Optional["MockDetector"] = None,
    ) -> None: ...

    def get_snapshot(self, cycle: int) -> "RealTelemetrySnapshot":
        """
        Fetch real telemetry for given cycle.

        Returns:
            RealTelemetrySnapshot with validation metadata attached

        Raises:
            TelemetryUnavailableError: If source cannot provide data (logged, not fatal)
        """
        ...

    def reset(self) -> None:
        """Reset adapter state for new run."""
        ...

    def get_validation_status(self) -> "ValidationStatus":
        """Get current rolling validation status."""
        ...

    def get_mock_detection_status(self) -> "MockDetectionStatus":
        """Get current mock detection status."""
        ...

    # SHADOW-MODE: These methods log but never control
    def _on_validation_failure(self, report: "ValidationReport") -> None:
        """Log validation failure. NO ABORT."""
        ...

    def _on_mock_detected(self, indicators: List["MockIndicator"]) -> None:
        """Log mock detection. NO ABORT."""
        ...
```

#### 1.2.3 TelemetryValidator

```python
@dataclass
class ValidationReport:
    """Per-window validation result."""
    window_start_cycle: int
    window_end_cycle: int
    is_valid: bool
    confidence: float  # [0.0, 1.0]
    violations: List[str]  # V1_BOUND_H, V2_VAR_H_LOW, etc.
    statistics: "ValidationStatistics"
    mock_indicators: "MockIndicatorSummary"
    timestamp_utc: str


@dataclass
class ValidationStatistics:
    """Statistical measures for validation."""
    var_H: float
    var_rho: float
    cor_H_rho: float
    acf_H_lag1: float
    kurtosis_H: float
    max_delta_H: float
    max_delta_rho: float
    unique_rho_count: int


class TelemetryValidator:
    """
    Implements RTTS V1-V6 validation checks.

    SHADOW MODE CONTRACT:
    - All validation is advisory
    - Failed validation logs TELEMETRY_VALIDATION_FAILED event
    - Never causes abort or control action
    """

    # RTTS Thresholds (Section 2.2)
    THRESHOLDS = {
        "var_H_min": 0.0001,
        "var_rho_min": 0.00005,
        "delta_H_max": 0.15,
        "delta_rho_max": 0.10,
        "cor_H_rho_min": 0.3,
        "cor_H_rho_max": 0.9,
        "acf_H_min": 0.05,
        "acf_H_max": 0.95,
        "unique_rho_min": 10,
    }

    def __init__(
        self,
        window_size: int = 200,
        min_confidence: float = 0.8,
    ) -> None: ...

    def add_snapshot(self, snapshot: "RealTelemetrySnapshot") -> Optional[ValidationReport]:
        """
        Add snapshot to validation window.

        Returns:
            ValidationReport if window completed, None otherwise
        """
        ...

    def validate_window(self, snapshots: List["RealTelemetrySnapshot"]) -> ValidationReport:
        """
        Validate a complete window of snapshots.

        Implements RTTS checks V1-V6:
        - V1: Boundedness (0 <= H, rho <= 1)
        - V2: Variance thresholds
        - V3: Continuity (jump detection)
        - V4: Correlation structure
        - V5: Temporal structure (autocorrelation)
        - V6: Value diversity
        """
        ...

    def _check_v1_boundedness(self, snapshots: List["RealTelemetrySnapshot"]) -> List[str]: ...
    def _check_v2_variance(self, snapshots: List["RealTelemetrySnapshot"]) -> List[str]: ...
    def _check_v3_continuity(self, snapshots: List["RealTelemetrySnapshot"]) -> List[str]: ...
    def _check_v4_correlation(self, snapshots: List["RealTelemetrySnapshot"]) -> List[str]: ...
    def _check_v5_autocorrelation(self, snapshots: List["RealTelemetrySnapshot"]) -> List[str]: ...
    def _check_v6_diversity(self, snapshots: List["RealTelemetrySnapshot"]) -> List[str]: ...

    def reset(self) -> None:
        """Reset validator state."""
        ...
```

#### 1.2.4 MockDetector

```python
@dataclass
class MockIndicator:
    """Single mock detection indicator."""
    indicator_id: str  # MOCK-001, MOCK-002, etc.
    severity: Literal["HIGH", "MEDIUM", "LOW"]
    triggered: bool
    value: float  # The measured value that triggered (or not)
    threshold: float  # The threshold that was checked
    description: str


@dataclass
class MockIndicatorSummary:
    """Summary of mock detection results."""
    high_severity_count: int
    medium_severity_count: int
    low_severity_count: int
    is_suspected_mock: bool
    indicators: List[MockIndicator]


class MockDetector:
    """
    Implements RTTS MOCK-001 through MOCK-010 detection.

    SHADOW MODE CONTRACT:
    - Detection is advisory only
    - SUSPECTED_MOCK status logged but never causes abort
    - Used for evidence generation and diagnostic review
    """

    # RTTS Mock Detection Criteria (Section 2.1)
    CRITERIA = {
        "MOCK-001": {"check": "var_H", "op": "<", "threshold": 0.0001, "severity": "HIGH"},
        "MOCK-002": {"check": "var_rho", "op": "<", "threshold": 0.00005, "severity": "HIGH"},
        "MOCK-003": {"check": "abs_cor_H_rho", "op": "<", "threshold": 0.1, "severity": "MEDIUM"},
        "MOCK-004": {"check": "abs_cor_H_rho", "op": ">", "threshold": 0.99, "severity": "MEDIUM"},
        "MOCK-005": {"check": "acf_H_lag1", "op": "<", "threshold": 0.05, "severity": "MEDIUM"},
        "MOCK-006": {"check": "acf_H_lag1", "op": ">", "threshold": 0.95, "severity": "MEDIUM"},
        "MOCK-007": {"check": "kurtosis_H", "op": "<", "threshold": -1.0, "severity": "LOW"},
        "MOCK-008": {"check": "kurtosis_H", "op": ">", "threshold": 5.0, "severity": "LOW"},
        "MOCK-009": {"check": "max_delta_H", "op": ">", "threshold": 0.15, "severity": "HIGH"},
        "MOCK-010": {"check": "unique_rho", "op": "<", "threshold": 10, "severity": "HIGH"},
    }

    def __init__(self) -> None: ...

    def detect(self, stats: ValidationStatistics) -> MockIndicatorSummary:
        """
        Run all mock detection criteria against statistics.

        Returns:
            MockIndicatorSummary with all indicator results
        """
        ...

    def _evaluate_criterion(
        self,
        criterion_id: str,
        stats: ValidationStatistics,
    ) -> MockIndicator: ...
```

### 1.3 Data Structures

#### 1.3.1 RealTelemetrySnapshot

```python
@dataclass
class RealTelemetrySnapshot:
    """
    Extended telemetry snapshot for real data.

    Extends base TelemetrySnapshot with real-only metadata.
    """
    # Base fields (from TelemetrySnapshot)
    cycle: int
    H: float
    rho: float
    tau: float
    beta: float
    omega: bool  # Safe region indicator

    # Real-only fields
    timestamp_utc: str
    source: Literal["REAL", "MOCK", "REPLAY"]
    instrumentation_id: str
    validation_status: Literal["PENDING", "VALIDATED", "FAILED", "UNVALIDATED"]
    validation_confidence: Optional[float] = None

    def to_base_snapshot(self) -> "TelemetrySnapshot":
        """Convert to base snapshot for compatibility."""
        ...
```

### 1.4 Shadow-Mode Invariants

#### 1.4.1 CANNOT Do (Hard Constraints)

```python
# These patterns are FORBIDDEN in P5 code:

# NO: Abort on validation failure
if not validation_report.is_valid:
    raise AbortError()  # FORBIDDEN

# NO: Throttle or rate-limit based on validation
if mock_detected:
    self.throttle_rate = 0.5  # FORBIDDEN

# NO: Emit control signals
governance_surface.emit_control_signal(...)  # FORBIDDEN

# NO: Modify USLA state
usla_integration.set_threshold(...)  # FORBIDDEN
```

#### 1.4.2 MUST Log (Evidence Requirements)

```python
# These events MUST be logged for evidence:

LOG_EVENTS = [
    "TELEMETRY_SNAPSHOT_RECEIVED",      # Every snapshot
    "VALIDATION_WINDOW_COMPLETED",       # Every validation window
    "VALIDATION_FAILED",                 # On validation failure
    "MOCK_INDICATOR_TRIGGERED",          # On any mock indicator
    "SUSPECTED_MOCK_DETECTED",           # On HIGH severity mock
    "TELEMETRY_UNAVAILABLE",             # On source failure
]

# Log format: JSONL to evidence pack
{
    "event": "VALIDATION_FAILED",
    "timestamp_utc": "2025-12-11T14:30:00Z",
    "cycle": 12345,
    "violations": ["V2_VAR_H_LOW", "V5_ACF_H"],
    "confidence": 0.6,
    "shadow_mode": True,  # ALWAYS True
    "action_taken": "LOGGED_ONLY"  # ALWAYS "LOGGED_ONLY"
}
```

### 1.5 Implementation Tasks

| Task ID | Description | Assignee | Dependencies |
|---------|-------------|----------|--------------|
| P5-IMPL-001 | Create `real_telemetry_adapter.py` module skeleton | `[CODEX]` | None |
| P5-IMPL-002 | Implement `RealTelemetryConfig` dataclass | `[CODEX]` | P5-IMPL-001 |
| P5-IMPL-003 | Implement `RealTelemetrySnapshot` dataclass | `[CODEX]` | P5-IMPL-001 |
| P5-IMPL-004 | Implement `ValidationStatistics` dataclass | `[CODEX]` | P5-IMPL-001 |
| P5-IMPL-005 | Implement `ValidationReport` dataclass | `[CODEX]` | P5-IMPL-004 |
| P5-IMPL-006 | Implement `MockIndicator` dataclass | `[CODEX]` | P5-IMPL-001 |
| P5-IMPL-007 | Implement `MockIndicatorSummary` dataclass | `[CODEX]` | P5-IMPL-006 |
| P5-IMPL-008 | Implement `TelemetryValidator._check_v1_boundedness` | `[CODEX]` | P5-IMPL-005 |
| P5-IMPL-009 | Implement `TelemetryValidator._check_v2_variance` | `[CODEX]` | P5-IMPL-005 |
| P5-IMPL-010 | Implement `TelemetryValidator._check_v3_continuity` | `[CODEX]` | P5-IMPL-005 |
| P5-IMPL-011 | Implement `TelemetryValidator._check_v4_correlation` | `[CODEX]` | P5-IMPL-005 |
| P5-IMPL-012 | Implement `TelemetryValidator._check_v5_autocorrelation` | `[CODEX]` | P5-IMPL-005 |
| P5-IMPL-013 | Implement `TelemetryValidator._check_v6_diversity` | `[CODEX]` | P5-IMPL-005 |
| P5-IMPL-014 | Implement `TelemetryValidator.validate_window` | `[CODEX]` | P5-IMPL-008..013 |
| P5-IMPL-015 | Implement `MockDetector.detect` | `[CODEX]` | P5-IMPL-007 |
| P5-IMPL-016 | Implement `RealTelemetryAdapter.get_snapshot` | `[CODEX]` | P5-IMPL-014, P5-IMPL-015 |
| P5-IMPL-017 | Write unit tests for TelemetryValidator | `[CODEX]` | P5-IMPL-014 |
| P5-IMPL-018 | Write unit tests for MockDetector | `[CODEX]` | P5-IMPL-015 |
| P5-IMPL-019 | `[ARCHITECT]` Review threshold values | `[ARCHITECT]` | P5-IMPL-017, P5-IMPL-018 |

---

## 2. DivergencePatternClassifier Blueprint

### 2.1 Module Location

```
backend/topology/first_light/divergence_patterns.py
```

### 2.2 Core Classes

#### 2.2.1 DivergencePattern Enum

```python
from enum import Enum, auto

class DivergencePattern(Enum):
    """RTTS divergence pattern taxonomy (Section 3.1)."""
    NONE = auto()           # No significant divergence
    DRIFT = auto()          # Systematic bias
    NOISE_AMPLIFICATION = auto()  # Twin over-sensitive
    PHASE_LAG = auto()      # Temporal misalignment
    ATTRACTOR_MISS = auto() # Safe region tracking failure
    TRANSIENT_MISS = auto() # Dynamic transition failure
    STRUCTURAL_BREAK = auto()  # Regime change detected
    UNCLASSIFIED = auto()   # Divergence present but no clear pattern
```

#### 2.2.2 DivergenceDecomposition

```python
@dataclass
class DivergenceDecomposition:
    """
    RTTS divergence decomposition (Section 3.2).

    Δ_total = Δ_bias + Δ_variance + Δ_timing + Δ_structural
    """
    delta_bias: float      # |mean(p_twin) - mean(p_real)|
    delta_variance: float  # |std(p_twin) - std(p_real)|
    delta_timing: float    # 1 - max(xcorr(p_twin, p_real))
    delta_structural: float  # rate(sign(Δp) changes)
    delta_total: float     # Sum of components

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "bias": round(self.delta_bias, 6),
            "variance": round(self.delta_variance, 6),
            "timing": round(self.delta_timing, 6),
            "structural": round(self.delta_structural, 6),
            "total": round(self.delta_total, 6),
        }
```

#### 2.2.3 PatternClassificationResult

```python
@dataclass
class PatternClassificationResult:
    """Result of divergence pattern classification."""
    pattern: DivergencePattern
    confidence: float  # [0.0, 1.0]
    decomposition: DivergenceDecomposition
    semantic_interpretation: str  # Human-readable explanation
    recommended_action: str  # Advisory only (SHADOW MODE)

    # Preserve existing severity semantics
    severity: Literal["NONE", "INFO", "WARN", "CRITICAL"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pattern": self.pattern.name,
            "confidence": round(self.confidence, 4),
            "decomposition": self.decomposition.to_dict(),
            "semantic_interpretation": self.semantic_interpretation,
            "recommended_action": self.recommended_action,
            "severity": self.severity,
        }
```

#### 2.2.4 DivergencePatternClassifier

```python
class DivergencePatternClassifier:
    """
    Classifies divergence patterns per RTTS Section 3.

    SHADOW MODE CONTRACT:
    - Classification is advisory only
    - recommended_action is never executed
    - Used for diagnostics and evidence generation
    """

    # RTTS Pattern Detection Thresholds (Section 3.4)
    THRESHOLDS = {
        "drift_mean_min": 0.05,
        "drift_std_max": 0.02,
        "noise_amp_factor": 2.0,
        "phase_lag_xcorr_min": 0.8,
        "structural_break_delta": 0.1,
        "structural_break_recent": 0.05,
        "transient_excursion_factor": 2.0,
        "transient_delta_ratio": 2.0,
    }

    # RTTS Semantic Mapping (Section 3.3)
    SEMANTIC_MAP = {
        DivergencePattern.NONE: "Twin model tracking within tolerance",
        DivergencePattern.DRIFT: "Twin model has calibration offset",
        DivergencePattern.NOISE_AMPLIFICATION: "Twin model is over-fitted to training noise",
        DivergencePattern.PHASE_LAG: "Twin model prediction horizon is misaligned",
        DivergencePattern.ATTRACTOR_MISS: "Twin model fundamentally misunderstands system dynamics",
        DivergencePattern.TRANSIENT_MISS: "Twin model lacks transient response fidelity",
        DivergencePattern.STRUCTURAL_BREAK: "Real system has undergone regime change",
        DivergencePattern.UNCLASSIFIED: "Divergence present but pattern unclear",
    }

    # RTTS Recommended Actions (Advisory Only)
    ACTION_MAP = {
        DivergencePattern.NONE: "Continue monitoring",
        DivergencePattern.DRIFT: "Recalibrate twin parameters",
        DivergencePattern.NOISE_AMPLIFICATION: "Increase twin smoothing",
        DivergencePattern.PHASE_LAG: "Adjust prediction horizon",
        DivergencePattern.ATTRACTOR_MISS: "Fundamental model review required",
        DivergencePattern.TRANSIENT_MISS: "Improve transient model",
        DivergencePattern.STRUCTURAL_BREAK: "Trigger re-initialization",
        DivergencePattern.UNCLASSIFIED: "Manual diagnostic review",
    }

    def __init__(self) -> None: ...

    def classify(
        self,
        real_series: List[float],
        twin_series: List[float],
        omega_real: Optional[List[bool]] = None,
        omega_twin: Optional[List[bool]] = None,
    ) -> PatternClassificationResult:
        """
        Classify divergence pattern from real vs twin series.

        Args:
            real_series: Real telemetry values (e.g., rho trajectory)
            twin_series: Twin model predictions
            omega_real: Real safe region indicators (optional)
            omega_twin: Twin safe region predictions (optional)

        Returns:
            PatternClassificationResult with pattern, decomposition, semantics
        """
        ...

    def compute_decomposition(
        self,
        real_series: List[float],
        twin_series: List[float],
    ) -> DivergenceDecomposition:
        """
        Compute RTTS divergence decomposition.

        Implements:
        - Δ_bias = |mean(p_twin) - mean(p_real)|
        - Δ_variance = |std(p_twin) - std(p_real)|
        - Δ_timing = 1 - max(xcorr(p_twin, p_real))
        - Δ_structural = rate(sign(Δp) changes)
        """
        ...

    def _detect_drift(
        self,
        delta: List[float],
        decomposition: DivergenceDecomposition,
    ) -> Tuple[bool, float]:
        """Check for DRIFT pattern. Returns (detected, confidence)."""
        ...

    def _detect_noise_amplification(
        self,
        real_series: List[float],
        twin_series: List[float],
        delta: List[float],
    ) -> Tuple[bool, float]:
        """Check for NOISE_AMPLIFICATION pattern."""
        ...

    def _detect_phase_lag(
        self,
        real_series: List[float],
        twin_series: List[float],
    ) -> Tuple[bool, float]:
        """Check for PHASE_LAG pattern."""
        ...

    def _detect_attractor_miss(
        self,
        omega_real: Optional[List[bool]],
        omega_twin: Optional[List[bool]],
    ) -> Tuple[bool, float]:
        """Check for ATTRACTOR_MISS pattern."""
        ...

    def _detect_transient_miss(
        self,
        real_series: List[float],
        delta: List[float],
    ) -> Tuple[bool, float]:
        """Check for TRANSIENT_MISS pattern."""
        ...

    def _detect_structural_break(
        self,
        delta: List[float],
    ) -> Tuple[bool, float]:
        """Check for STRUCTURAL_BREAK pattern."""
        ...

    def _map_to_severity(
        self,
        pattern: DivergencePattern,
        decomposition: DivergenceDecomposition,
    ) -> Literal["NONE", "INFO", "WARN", "CRITICAL"]:
        """
        Map pattern to existing severity semantics.

        Preserves compatibility with P4 DivergenceAnalyzer:
        - NONE: delta_total < 0.01
        - INFO: delta_total < 0.05
        - WARN: delta_total < 0.15
        - CRITICAL: delta_total >= 0.15 OR STRUCTURAL_BREAK
        """
        ...
```

### 2.3 Integration with P4 DivergenceAnalyzer

#### 2.3.1 New Fields for DivergenceSnapshot

```python
# Extend existing DivergenceSnapshot (backend/topology/first_light/divergence_analyzer.py)

@dataclass
class DivergenceSnapshot:
    """Extended with P5 pattern classification."""
    # Existing fields
    cycle: int
    delta_p: float
    delta_p_pct: float
    severity: str
    real_state: Dict[str, Any]
    twin_state: Dict[str, Any]
    telemetry_hash: str

    # NEW P5 fields
    pattern_classification: Optional[PatternClassificationResult] = None
    decomposition: Optional[DivergenceDecomposition] = None
```

#### 2.3.2 Integration Hook

```python
# In DivergenceAnalyzer.analyze() or similar:

def analyze_with_patterns(
    self,
    real_trajectory: List[TelemetrySnapshot],
    twin_trajectory: List[TelemetrySnapshot],
    window_size: int = 50,
) -> List[DivergenceSnapshot]:
    """
    Enhanced analysis with pattern classification.

    Calls DivergencePatternClassifier at window boundaries.
    """
    classifier = DivergencePatternClassifier()

    # ... existing divergence computation ...

    # At window boundaries, classify pattern
    if cycle % window_size == 0 and cycle > 0:
        window_real = [s.rho for s in real_trajectory[cycle-window_size:cycle]]
        window_twin = [s.rho for s in twin_trajectory[cycle-window_size:cycle]]

        classification = classifier.classify(window_real, window_twin)

        # Attach to snapshot
        snapshot.pattern_classification = classification
        snapshot.decomposition = classification.decomposition

    return snapshots
```

### 2.4 Test Plan

#### 2.4.1 Synthetic Test Cases

```python
# tests/topology/test_divergence_patterns.py

class TestDivergencePatternClassifier:
    """Test suite with synthetic examples for each pattern."""

    def test_pattern_none(self):
        """Twin tracks real within tolerance."""
        real = [0.5 + 0.01 * random() for _ in range(100)]
        twin = [r + 0.005 * random() for r in real]  # Tiny divergence
        result = classifier.classify(real, twin)
        assert result.pattern == DivergencePattern.NONE
        assert result.severity == "NONE"

    def test_pattern_drift(self):
        """Systematic offset between twin and real."""
        real = [0.5 + 0.01 * sin(i/10) for i in range(100)]
        twin = [r + 0.08 for r in real]  # Constant 0.08 offset
        result = classifier.classify(real, twin)
        assert result.pattern == DivergencePattern.DRIFT
        assert result.decomposition.delta_bias > 0.05

    def test_pattern_noise_amplification(self):
        """Twin has excessive variance."""
        real = [0.5 + 0.01 * random() for _ in range(100)]
        twin = [0.5 + 0.05 * random() for _ in range(100)]  # 5x noise
        result = classifier.classify(real, twin)
        assert result.pattern == DivergencePattern.NOISE_AMPLIFICATION

    def test_pattern_phase_lag(self):
        """Twin is temporally shifted."""
        real = [0.5 + 0.1 * sin(i/5) for i in range(100)]
        twin = [0.5 + 0.1 * sin((i-3)/5) for i in range(100)]  # 3-cycle lag
        result = classifier.classify(real, twin)
        assert result.pattern == DivergencePattern.PHASE_LAG

    def test_pattern_attractor_miss(self):
        """Twin fails to track safe region."""
        omega_real = [True] * 80 + [False] * 20
        omega_twin = [True] * 50 + [False] * 50  # Disagrees on 30 cycles
        real = [0.7 if o else 0.3 for o in omega_real]
        twin = [0.7 if o else 0.3 for o in omega_twin]
        result = classifier.classify(real, twin, omega_real, omega_twin)
        assert result.pattern == DivergencePattern.ATTRACTOR_MISS

    def test_pattern_transient_miss(self):
        """Twin misses excursions only."""
        real = [0.5] * 40 + [0.9, 0.95, 0.85, 0.5] + [0.5] * 56  # Spike at 40-43
        twin = [0.5] * 100  # Flat, misses the spike
        result = classifier.classify(real, twin)
        assert result.pattern == DivergencePattern.TRANSIENT_MISS

    def test_pattern_structural_break(self):
        """Sudden persistent divergence."""
        real = [0.5] * 50 + [0.8] * 50  # Step change at 50
        twin = [0.5] * 100  # Doesn't adapt
        result = classifier.classify(real, twin)
        assert result.pattern == DivergencePattern.STRUCTURAL_BREAK
        assert result.severity == "CRITICAL"
```

### 2.5 Implementation Tasks

| Task ID | Description | Assignee | Dependencies |
|---------|-------------|----------|--------------|
| P5-IMPL-020 | Create `divergence_patterns.py` module | `[CODEX]` | None |
| P5-IMPL-021 | Implement `DivergencePattern` enum | `[CODEX]` | P5-IMPL-020 |
| P5-IMPL-022 | Implement `DivergenceDecomposition` dataclass | `[CODEX]` | P5-IMPL-020 |
| P5-IMPL-023 | Implement `PatternClassificationResult` dataclass | `[CODEX]` | P5-IMPL-021, P5-IMPL-022 |
| P5-IMPL-024 | Implement `DivergencePatternClassifier.compute_decomposition` | `[CODEX]` | P5-IMPL-022 |
| P5-IMPL-025 | Implement `_detect_drift` | `[CODEX]` | P5-IMPL-024 |
| P5-IMPL-026 | Implement `_detect_noise_amplification` | `[CODEX]` | P5-IMPL-024 |
| P5-IMPL-027 | Implement `_detect_phase_lag` | `[CODEX]` | P5-IMPL-024 |
| P5-IMPL-028 | Implement `_detect_attractor_miss` | `[CODEX]` | P5-IMPL-024 |
| P5-IMPL-029 | Implement `_detect_transient_miss` | `[CODEX]` | P5-IMPL-024 |
| P5-IMPL-030 | Implement `_detect_structural_break` | `[CODEX]` | P5-IMPL-024 |
| P5-IMPL-031 | Implement `DivergencePatternClassifier.classify` | `[CODEX]` | P5-IMPL-025..030 |
| P5-IMPL-032 | Implement `_map_to_severity` | `[CODEX]` | P5-IMPL-031 |
| P5-IMPL-033 | Write test suite with 7 pattern cases | `[CODEX]` | P5-IMPL-031 |
| P5-IMPL-034 | Extend `DivergenceSnapshot` with P5 fields | `[CODEX]` | P5-IMPL-023 |
| P5-IMPL-035 | Integrate classifier into `DivergenceAnalyzer` | `[CODEX]` | P5-IMPL-031, P5-IMPL-034 |
| P5-IMPL-036 | `[ARCHITECT]` Review pattern detection thresholds | `[ARCHITECT]` | P5-IMPL-033 |

---

## 3. P5AcceptanceGate Design

### 3.1 Module Location

```
backend/health/p5_acceptance_gate.py
```

### 3.2 Core Classes

#### 3.2.1 GateStatus

```python
@dataclass
class GateCheckResult:
    """Result of a single acceptance criterion check."""
    criterion_id: str  # P5-TV-001, P5-CAL-002, etc.
    description: str
    passed: bool
    actual_value: Any
    threshold: Any
    evidence_ref: Optional[str] = None  # Path to supporting evidence


@dataclass
class GateStatus:
    """Status of a single acceptance gate."""
    gate_name: str  # "telemetry_validation", "calibration", etc.
    passed: bool
    checks: List[GateCheckResult]
    failure_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate": self.gate_name,
            "passed": self.passed,
            "failure_count": self.failure_count,
            "checks": [
                {
                    "id": c.criterion_id,
                    "passed": c.passed,
                    "actual": c.actual_value,
                    "threshold": c.threshold,
                }
                for c in self.checks
            ],
        }
```

#### 3.2.2 P5AcceptanceGateSummary

```python
@dataclass
class P5AcceptanceGateSummary:
    """
    Complete P5 acceptance gate summary.

    SHADOW MODE CONTRACT:
    - verdict is ADVISORY ONLY
    - Never used for control decisions
    - Always logged to evidence pack
    """
    timestamp_utc: str
    run_id: str

    # Per-gate status
    telemetry_gate: GateStatus
    calibration_gate: GateStatus
    divergence_gate: GateStatus
    operational_gate: GateStatus

    # Overall verdict (SHADOW-only, advisory)
    verdict: Literal["ACCEPT", "WARNING", "REJECT"]
    verdict_rationale: str

    # Evidence references
    evidence_pack_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "run_id": self.run_id,
            "gates": {
                "telemetry_validation": self.telemetry_gate.to_dict(),
                "calibration": self.calibration_gate.to_dict(),
                "divergence": self.divergence_gate.to_dict(),
                "operational": self.operational_gate.to_dict(),
            },
            "verdict": self.verdict,
            "verdict_rationale": self.verdict_rationale,
            "shadow_mode": True,  # ALWAYS True
            "advisory_only": True,  # ALWAYS True
        }
```

#### 3.2.3 P5AcceptanceGate

```python
class P5AcceptanceGate:
    """
    P5 acceptance gate implementing RTTS Section 5 criteria.

    SHADOW MODE CONTRACT:
    - All evaluations are advisory only
    - verdict field is never used for control
    - Results logged to evidence pack for human review
    """

    # RTTS Acceptance Thresholds (Section 5.2)
    THRESHOLDS = {
        # Telemetry Validation Gate
        "P5-TV-001": {"name": "validation_confidence_min", "value": 0.90},
        "P5-TV-002": {"name": "mock_high_severity_max", "value": 0},
        "P5-TV-003": {"name": "validation_pass_rate_min", "value": 0.95},

        # Calibration Gate
        "P5-CAL-001": {"name": "divergence_converged_max", "value": 0.05},
        "P5-CAL-002": {"name": "drift_24h_max", "value": 0.02},
        "P5-CAL-003": {"name": "recalibration_triggers_max", "value": 0},

        # Divergence Gate
        "P5-DIV-001": {"name": "mean_divergence_max", "value": 0.05},
        "P5-DIV-002": {"name": "max_divergence_max", "value": 0.15},
        "P5-DIV-003": {"name": "std_divergence_max", "value": 0.03},
        "P5-DIV-004": {"name": "critical_events_max", "value": 0},
        "P5-DIV-005": {"name": "structural_break_max", "value": 0},

        # Operational Gate
        "P5-OPS-001": {"name": "shadow_mode_percentage", "value": 100.0},
        "P5-OPS-002": {"name": "governance_interventions_max", "value": 0},
        "P5-OPS-003": {"name": "schema_validation_pass", "value": True},
        "P5-OPS-004": {"name": "human_review_completed", "value": True},
    }

    def __init__(self) -> None: ...

    def evaluate(
        self,
        validation_reports: List[ValidationReport],
        divergence_summary: "DivergenceSummary",
        calibration_summary: "CalibrationSummary",
        operational_status: "OperationalStatus",
    ) -> P5AcceptanceGateSummary:
        """
        Evaluate all P5 acceptance criteria.

        Returns:
            P5AcceptanceGateSummary with per-gate status and verdict
        """
        ...

    def _evaluate_telemetry_gate(
        self,
        validation_reports: List[ValidationReport],
    ) -> GateStatus:
        """Evaluate P5-TV-001 through P5-TV-003."""
        ...

    def _evaluate_calibration_gate(
        self,
        calibration_summary: "CalibrationSummary",
    ) -> GateStatus:
        """Evaluate P5-CAL-001 through P5-CAL-003."""
        ...

    def _evaluate_divergence_gate(
        self,
        divergence_summary: "DivergenceSummary",
    ) -> GateStatus:
        """Evaluate P5-DIV-001 through P5-DIV-005."""
        ...

    def _evaluate_operational_gate(
        self,
        operational_status: "OperationalStatus",
    ) -> GateStatus:
        """Evaluate P5-OPS-001 through P5-OPS-004."""
        ...

    def _compute_verdict(
        self,
        gates: List[GateStatus],
    ) -> Tuple[str, str]:
        """
        Compute overall verdict (ADVISORY ONLY).

        Returns:
            (verdict, rationale) tuple

        Logic:
        - ACCEPT: All gates pass
        - WARNING: Only OPS gate failures OR only LOW severity failures
        - REJECT: Any TV/CAL/DIV gate failure with HIGH severity
        """
        ...
```

### 3.3 Integration Points

#### 3.3.1 Global Surface Attachment

```python
# In backend/health/global_surface.py or similar:

class GlobalHealthSurface:
    """Global health monitoring surface."""

    def __init__(self):
        self.tiles: Dict[str, Any] = {}

    def attach_p5_acceptance_tile(
        self,
        gate: P5AcceptanceGate,
    ) -> None:
        """
        Attach P5 acceptance gate as SHADOW tile.

        SHADOW CONTRACT:
        - Tile is read-only
        - Never influences governance decisions
        - Updated at run boundaries only
        """
        self.tiles["p5_acceptance"] = {
            "type": "shadow",
            "gate": gate,
            "last_summary": None,
            "update_frequency": "per_run",
        }

    def update_p5_tile(
        self,
        summary: P5AcceptanceGateSummary,
    ) -> None:
        """Update P5 tile with latest summary."""
        if "p5_acceptance" in self.tiles:
            self.tiles["p5_acceptance"]["last_summary"] = summary
```

#### 3.3.2 Evidence Pack Integration

```python
# Evidence pack structure:
evidence_pack = {
    "metadata": {...},
    "governance": {
        "p5_acceptance": {
            "summary": p5_summary.to_dict(),
            "gate_reports": {
                "telemetry": [...validation_reports...],
                "calibration": calibration_report,
                "divergence": divergence_report,
                "operational": operational_report,
            },
            "schema_version": "1.0.0",
        },
    },
    # ... other evidence ...
}
```

### 3.4 Implementation Tasks

| Task ID | Description | Assignee | Dependencies |
|---------|-------------|----------|--------------|
| P5-IMPL-037 | Create `p5_acceptance_gate.py` module | `[CODEX]` | None |
| P5-IMPL-038 | Implement `GateCheckResult` dataclass | `[CODEX]` | P5-IMPL-037 |
| P5-IMPL-039 | Implement `GateStatus` dataclass | `[CODEX]` | P5-IMPL-038 |
| P5-IMPL-040 | Implement `P5AcceptanceGateSummary` dataclass | `[CODEX]` | P5-IMPL-039 |
| P5-IMPL-041 | Implement `_evaluate_telemetry_gate` | `[CODEX]` | P5-IMPL-039 |
| P5-IMPL-042 | Implement `_evaluate_calibration_gate` | `[CODEX]` | P5-IMPL-039 |
| P5-IMPL-043 | Implement `_evaluate_divergence_gate` | `[CODEX]` | P5-IMPL-039 |
| P5-IMPL-044 | Implement `_evaluate_operational_gate` | `[CODEX]` | P5-IMPL-039 |
| P5-IMPL-045 | Implement `_compute_verdict` | `[CODEX]` | P5-IMPL-041..044 |
| P5-IMPL-046 | Implement `P5AcceptanceGate.evaluate` | `[CODEX]` | P5-IMPL-045 |
| P5-IMPL-047 | Write unit tests for each gate | `[CODEX]` | P5-IMPL-046 |
| P5-IMPL-048 | Implement global surface attachment | `[CODEX]` | P5-IMPL-046 |
| P5-IMPL-049 | Implement evidence pack integration | `[CODEX]` | P5-IMPL-046 |
| P5-IMPL-050 | `[ARCHITECT]` Review verdict logic | `[ARCHITECT]` | P5-IMPL-047 |

---

## 4. Twin Warm-Start Calibration Implementation Plan

### 4.1 Module Location

```
backend/topology/first_light/twin_calibration.py
```

Extends existing `TwinRunner` in `runner_p4.py` rather than replacing it.

### 4.2 Core Classes

#### 4.2.1 CalibrationConfig

```python
@dataclass
class CalibrationConfig:
    """Configuration for twin calibration."""
    # Phase 1: Historical Alignment
    min_history_cycles: int = 1000
    history_source: Literal["file", "database", "live_buffer"]
    history_path: Optional[str] = None

    # Phase 2: Parameter Optimization
    max_iterations: int = 100
    convergence_threshold: float = 0.05  # ε_calibration
    perturbation_scale_initial: float = 0.01
    perturbation_decay: float = 0.99

    # Phase 3: Online Validation
    validation_window_size: int = 200
    recalibration_trigger_threshold: float = 0.10
    recalibration_trigger_windows: int = 3
```

#### 4.2.2 CalibrationResult

```python
@dataclass
class CalibrationResult:
    """Result of calibration attempt."""
    converged: bool
    final_divergence: float
    iterations: int
    initial_params: Dict[str, float]
    final_params: Dict[str, float]
    history_cycles_used: int
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "converged": self.converged,
            "final_divergence": round(self.final_divergence, 6),
            "iterations": self.iterations,
            "params_delta": {
                k: round(self.final_params[k] - self.initial_params[k], 6)
                for k in self.final_params
            },
            "history_cycles": self.history_cycles_used,
            "timestamp_utc": self.timestamp_utc,
        }
```

#### 4.2.3 CalibrationSummary

```python
@dataclass
class CalibrationSummary:
    """Summary for P5 acceptance gate."""
    last_calibration: CalibrationResult
    drift_since_calibration: float
    recalibration_trigger_count: int
    is_stable: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "last_calibration": self.last_calibration.to_dict(),
            "drift_since_calibration": round(self.drift_since_calibration, 6),
            "recalibration_triggers": self.recalibration_trigger_count,
            "is_stable": self.is_stable,
        }
```

#### 4.2.4 TwinCalibrator

```python
class TwinCalibrator:
    """
    Implements RTTS Section 4 twin warm-start calibration.

    SHADOW MODE CONTRACT:
    - Calibration modifies twin model only
    - Never affects real system state
    - All results logged for evidence
    """

    def __init__(
        self,
        config: CalibrationConfig,
        twin_runner: "TwinRunner",
    ) -> None:
        self.config = config
        self.twin = twin_runner
        self._calibration_history: List[CalibrationResult] = []
        self._online_divergence_buffer: List[float] = []

    # Phase 1: Historical Alignment
    def load_history(
        self,
        source: Optional[str] = None,
    ) -> List["RealTelemetrySnapshot"]:
        """
        Load historical telemetry for calibration.

        Returns:
            List of validated historical snapshots

        Raises:
            CalibrationError: If insufficient history available
        """
        ...

    # Phase 2: Parameter Optimization
    def calibrate(
        self,
        history: List["RealTelemetrySnapshot"],
    ) -> CalibrationResult:
        """
        Optimize twin parameters against history.

        Implements gradient-free optimization:
        1. Run twin over history
        2. Compute divergence
        3. Perturb parameters
        4. Repeat until convergence or max iterations

        Returns:
            CalibrationResult with final parameters
        """
        ...

    def _compute_history_divergence(
        self,
        history: List["RealTelemetrySnapshot"],
    ) -> float:
        """Run twin over history and compute mean absolute divergence."""
        ...

    def _perturb_twin_params(
        self,
        scale: float,
    ) -> Dict[str, float]:
        """Perturb twin parameters by scale factor."""
        ...

    # Phase 3: Online Validation
    def update_online(
        self,
        real_snapshot: "RealTelemetrySnapshot",
        twin_snapshot: "TelemetrySnapshot",
    ) -> Optional[str]:
        """
        Update online validation state.

        Returns:
            "RECALIBRATE" if trigger condition met, None otherwise

        SHADOW MODE: Returns advisory string, never auto-triggers
        """
        ...

    def check_recalibration_trigger(self) -> bool:
        """
        Check if recalibration should be triggered.

        RTTS criteria:
        - Rolling mean(|Δp|) > 0.10 for 3 consecutive windows
        """
        ...

    def get_summary(self) -> CalibrationSummary:
        """Get current calibration summary for P5 gate."""
        ...
```

### 4.3 API Integration

#### 4.3.1 Offline Calibration API

```python
# Usage in calibration script or harness:

def run_offline_calibration(
    history_path: str,
    twin_config: TwinConfig,
) -> CalibrationResult:
    """
    Run offline calibration before P5 live wire.

    Args:
        history_path: Path to historical telemetry JSONL
        twin_config: Twin model configuration

    Returns:
        CalibrationResult with optimized parameters
    """
    config = CalibrationConfig(
        history_source="file",
        history_path=history_path,
    )

    twin = TwinRunner(twin_config)
    calibrator = TwinCalibrator(config, twin)

    history = calibrator.load_history()
    result = calibrator.calibrate(history)

    if not result.converged:
        logging.warning(f"Calibration did not converge: {result.final_divergence}")

    return result
```

#### 4.3.2 Online Validation Hook

```python
# Integration with FirstLightShadowRunnerP4:

class FirstLightShadowRunnerP4:
    def __init__(self, ...):
        ...
        self.calibrator: Optional[TwinCalibrator] = None

    def attach_calibrator(
        self,
        calibrator: TwinCalibrator,
    ) -> None:
        """Attach calibrator for online validation."""
        self.calibrator = calibrator

    def _on_cycle_complete(
        self,
        real_snapshot: RealTelemetrySnapshot,
        twin_snapshot: TelemetrySnapshot,
    ) -> None:
        """Hook called after each cycle."""
        if self.calibrator:
            trigger = self.calibrator.update_online(real_snapshot, twin_snapshot)
            if trigger == "RECALIBRATE":
                # SHADOW MODE: Log only, never auto-recalibrate
                logging.info("Recalibration trigger detected (advisory)")
                self._log_event("RECALIBRATION_TRIGGER_ADVISORY")
```

### 4.4 Test Harness Plan

#### 4.4.1 Synthetic Calibration Dataset

```python
# tests/topology/fixtures/calibration_dataset.py

def generate_calibration_dataset(
    cycles: int = 1000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic dataset for calibration testing.

    Properties:
    - Realistic USLA dynamics (bounded, continuous, correlated)
    - Known ground truth parameters for validation
    - Includes transient events for robustness testing
    """
    rng = random.Random(seed)

    # Ground truth parameters
    H_base = 0.7
    rho_base = 0.65
    coupling = 0.5  # H-rho coupling strength
    noise_scale = 0.02

    dataset = []
    H_prev, rho_prev = H_base, rho_base

    for cycle in range(cycles):
        # Coupled dynamics
        H_new = H_prev + coupling * (rho_prev - rho_base) + noise_scale * rng.gauss(0, 1)
        rho_new = rho_prev + coupling * (H_prev - H_base) + noise_scale * rng.gauss(0, 1)

        # Bound to [0, 1]
        H_new = max(0, min(1, H_new))
        rho_new = max(0, min(1, rho_new))

        # Occasional transient
        if cycle == 400:
            H_new = 0.95  # Spike

        dataset.append({
            "cycle": cycle,
            "H": H_new,
            "rho": rho_new,
            "tau": 0.5,
            "beta": 0.1,
            "omega": rho_new > 0.5,
        })

        H_prev, rho_prev = H_new, rho_new

    return dataset
```

#### 4.4.2 Calibration Success Criteria

```python
# tests/topology/test_twin_calibration.py

class TestTwinCalibration:
    """Test suite for twin calibration."""

    def test_calibration_converges(self):
        """Calibration converges on synthetic data."""
        dataset = generate_calibration_dataset(cycles=1000)
        result = run_offline_calibration(dataset)

        assert result.converged
        assert result.final_divergence < 0.05
        assert result.iterations < 100

    def test_calibration_reduces_divergence(self):
        """Calibration reduces divergence vs uncalibrated."""
        dataset = generate_calibration_dataset(cycles=1000)

        # Measure uncalibrated divergence
        twin_uncalibrated = TwinRunner(default_config())
        uncalibrated_div = compute_divergence(twin_uncalibrated, dataset)

        # Calibrate
        result = run_offline_calibration(dataset)

        # Measure calibrated divergence
        calibrated_div = result.final_divergence

        assert calibrated_div < uncalibrated_div * 0.5  # At least 50% reduction

    def test_recalibration_trigger_detection(self):
        """Recalibration trigger fires when divergence exceeds threshold."""
        calibrator = TwinCalibrator(config, twin)

        # Simulate 3 windows of high divergence
        for window in range(3):
            for cycle in range(50):
                calibrator.update_online(
                    real_snapshot=make_snapshot(rho=0.7),
                    twin_snapshot=make_snapshot(rho=0.5),  # 0.2 divergence
                )

        assert calibrator.check_recalibration_trigger()

    def test_calibration_shadow_mode(self):
        """Calibration never modifies real system state."""
        # Verify no control signals emitted
        # Verify no writes to USLA integration
        # Verify all outputs are logged only
        ...
```

### 4.5 Implementation Tasks

| Task ID | Description | Assignee | Dependencies |
|---------|-------------|----------|--------------|
| P5-IMPL-051 | Create `twin_calibration.py` module | `[CODEX]` | None |
| P5-IMPL-052 | Implement `CalibrationConfig` dataclass | `[CODEX]` | P5-IMPL-051 |
| P5-IMPL-053 | Implement `CalibrationResult` dataclass | `[CODEX]` | P5-IMPL-051 |
| P5-IMPL-054 | Implement `CalibrationSummary` dataclass | `[CODEX]` | P5-IMPL-053 |
| P5-IMPL-055 | Implement `TwinCalibrator.load_history` | `[CODEX]` | P5-IMPL-052 |
| P5-IMPL-056 | Implement `TwinCalibrator._compute_history_divergence` | `[CODEX]` | P5-IMPL-055 |
| P5-IMPL-057 | Implement `TwinCalibrator._perturb_twin_params` | `[CODEX]` | P5-IMPL-055 |
| P5-IMPL-058 | Implement `TwinCalibrator.calibrate` | `[CODEX]` | P5-IMPL-056, P5-IMPL-057 |
| P5-IMPL-059 | Implement `TwinCalibrator.update_online` | `[CODEX]` | P5-IMPL-054 |
| P5-IMPL-060 | Implement `TwinCalibrator.check_recalibration_trigger` | `[CODEX]` | P5-IMPL-059 |
| P5-IMPL-061 | Generate synthetic calibration dataset | `[CODEX]` | None |
| P5-IMPL-062 | Write calibration convergence tests | `[CODEX]` | P5-IMPL-058, P5-IMPL-061 |
| P5-IMPL-063 | Write recalibration trigger tests | `[CODEX]` | P5-IMPL-060 |
| P5-IMPL-064 | Integrate calibrator with FirstLightShadowRunnerP4 | `[CODEX]` | P5-IMPL-059 |
| P5-IMPL-065 | `[ARCHITECT]` Review optimization parameters | `[ARCHITECT]` | P5-IMPL-062 |

---

## 5. P5 Live Wire Run Plan

### 5.1 Pre-Flight Checklist

All items must be GREEN before P5 Live Wire execution.

#### 5.1.1 Module Readiness

| Check ID | Description | Verification |
|----------|-------------|--------------|
| PRE-001 | `real_telemetry_adapter.py` exists and imports | `python -c "from backend.topology.first_light.real_telemetry_adapter import *"` |
| PRE-002 | `divergence_patterns.py` exists and imports | `python -c "from backend.topology.first_light.divergence_patterns import *"` |
| PRE-003 | `p5_acceptance_gate.py` exists and imports | `python -c "from backend.health.p5_acceptance_gate import *"` |
| PRE-004 | `twin_calibration.py` exists and imports | `python -c "from backend.topology.first_light.twin_calibration import *"` |

#### 5.1.2 Test Suite Readiness

| Check ID | Description | Verification |
|----------|-------------|--------------|
| PRE-005 | TelemetryValidator tests pass | `pytest tests/topology/test_telemetry_validator.py -v` |
| PRE-006 | MockDetector tests pass | `pytest tests/topology/test_mock_detector.py -v` |
| PRE-007 | DivergencePatternClassifier tests pass | `pytest tests/topology/test_divergence_patterns.py -v` |
| PRE-008 | P5AcceptanceGate tests pass | `pytest tests/health/test_p5_acceptance_gate.py -v` |
| PRE-009 | TwinCalibrator tests pass | `pytest tests/topology/test_twin_calibration.py -v` |
| PRE-010 | All P5 tests pass | `pytest -k "p5" -v` |

#### 5.1.3 Calibration Readiness

| Check ID | Description | Verification |
|----------|-------------|--------------|
| PRE-011 | Historical telemetry available | Minimum 1000 cycles in calibration dataset |
| PRE-012 | Offline calibration completed | `CalibrationResult.converged == True` |
| PRE-013 | Calibration divergence < 0.05 | `CalibrationResult.final_divergence < 0.05` |

#### 5.1.4 Infrastructure Readiness

| Check ID | Description | Verification |
|----------|-------------|--------------|
| PRE-014 | Output directory exists | `mkdir -p artifacts/p5_live_wire/` |
| PRE-015 | Logging configured | JSONL output to `artifacts/p5_live_wire/` |
| PRE-016 | Evidence pack schema valid | JSON schema validation passes |

### 5.2 Run Parameters

```yaml
# P5 Live Wire Run Configuration
run:
  id: "p5-live-wire-001"
  cycles: 500
  seed: null  # No seed for real telemetry

telemetry:
  source_type: "usla_live"  # Or "usla_replay" for first attempt
  instrumentation_id: "usla-prod-001"
  validation_window_size: 200
  validation_frequency: 50

twin:
  # Use calibrated parameters from PRE-012
  use_calibrated_params: true
  calibration_result_path: "artifacts/calibration/latest.json"

divergence:
  pattern_classification_enabled: true
  window_size: 50

acceptance:
  evaluate_at_end: true
  generate_evidence_pack: true

logging:
  output_dir: "artifacts/p5_live_wire/"
  log_level: "INFO"
  jsonl_enabled: true
  artifacts:
    - "validation_reports.jsonl"
    - "divergence_snapshots.jsonl"
    - "pattern_classifications.jsonl"
    - "calibration_online.jsonl"
    - "p5_acceptance_summary.json"
```

### 5.3 Metrics to Monitor (Priority Order)

#### 5.3.1 Telemetry Validation (First Priority)

```
WATCH:
  - validation_confidence: Should be >= 0.90
  - mock_high_severity_count: Should be 0
  - validation_pass_rate: Should be >= 0.95

RED FLAGS:
  - validation_confidence < 0.80 → Telemetry source issue
  - mock_high_severity_count > 0 → Possible test data leakage
  - Frequent V3_JUMP violations → Instrumentation instability
```

#### 5.3.2 Divergence Pattern (Second Priority)

```
WATCH:
  - pattern_distribution: Expect mostly NONE, some DRIFT acceptable
  - mean_divergence: Should be < 0.05
  - max_divergence: Should be < 0.15

RED FLAGS:
  - STRUCTURAL_BREAK detected → Regime change, abort analysis
  - ATTRACTOR_MISS frequent → Twin model fundamentally wrong
  - mean_divergence > 0.10 → Calibration failed
```

#### 5.3.3 Acceptance Envelope (Third Priority)

```
WATCH:
  - Per-gate pass/fail status
  - Overall verdict (ACCEPT/WARNING/REJECT)
  - Specific failing criteria

RED FLAGS:
  - Any CRITICAL severity divergence event
  - Recalibration trigger fired
  - Operational gate failures (should be impossible in shadow mode)
```

### 5.4 Artifact Generation

#### 5.4.1 Required Artifacts

| Artifact | Format | Description |
|----------|--------|-------------|
| `p5_run_manifest.json` | JSON | Run configuration and parameters |
| `validation_reports.jsonl` | JSONL | Per-window validation reports |
| `divergence_snapshots.jsonl` | JSONL | Per-cycle divergence with patterns |
| `pattern_classifications.jsonl` | JSONL | Per-window pattern classifications |
| `calibration_online.jsonl` | JSONL | Online calibration state |
| `p5_acceptance_summary.json` | JSON | Final acceptance gate summary |
| `evidence_pack_p5.json` | JSON | Complete evidence bundle |

#### 5.4.2 Evidence Pack Structure

```json
{
  "metadata": {
    "run_id": "p5-live-wire-001",
    "timestamp_utc": "2025-12-11T15:00:00Z",
    "cycles": 500,
    "phase": "P5",
    "shadow_mode": true
  },
  "telemetry": {
    "validation_reports": [...],
    "mock_detection_summary": {...},
    "aggregate_statistics": {...}
  },
  "divergence": {
    "snapshots": [...],
    "pattern_summary": {
      "NONE": 8,
      "DRIFT": 2,
      "STRUCTURAL_BREAK": 0
    },
    "severity_distribution": {...}
  },
  "calibration": {
    "initial_result": {...},
    "online_history": [...],
    "recalibration_triggers": 0
  },
  "governance": {
    "p5_acceptance": {
      "summary": {...},
      "verdict": "ACCEPT",
      "advisory_only": true
    }
  },
  "artifacts": {
    "validation_reports_path": "validation_reports.jsonl",
    "divergence_snapshots_path": "divergence_snapshots.jsonl"
  }
}
```

### 5.5 Post-Run Analysis Protocol

```
1. IMMEDIATE (within 1 hour):
   - Review p5_acceptance_summary.json verdict
   - Check for any CRITICAL severity events
   - Verify shadow_mode was maintained (should be 100%)

2. DETAILED (within 24 hours):
   - Analyze pattern_classifications distribution
   - Compare validation_confidence trajectory to RTTS thresholds
   - Review any WARNING-level criteria that failed

3. ARCHIVE:
   - Bundle all artifacts into evidence pack
   - Commit evidence_pack_p5.json to repo
   - Update Phase_X_P3P4_TODO.md with P5 checkpoints
```

### 5.6 Live Wire Execution Tasks

| Task ID | Description | Assignee | Dependencies |
|---------|-------------|----------|--------------|
| P5-IMPL-066 | Create P5 live wire harness script | `[CODEX]` | All P5-IMPL-001..065 |
| P5-IMPL-067 | Implement pre-flight checklist automation | `[CODEX]` | P5-IMPL-066 |
| P5-IMPL-068 | Implement artifact generation | `[CODEX]` | P5-IMPL-066 |
| P5-IMPL-069 | Implement evidence pack bundler | `[CODEX]` | P5-IMPL-068 |
| P5-IMPL-070 | Write post-run analysis report generator | `[CODEX]` | P5-IMPL-069 |
| P5-IMPL-071 | `[ARCHITECT]` Conduct pre-flight review | `[ARCHITECT]` | P5-IMPL-067 |
| P5-IMPL-072 | `[ARCHITECT]` Execute P5 Live Wire run | `[ARCHITECT]` | P5-IMPL-071 |
| P5-IMPL-073 | `[ARCHITECT]` Review post-run evidence pack | `[ARCHITECT]` | P5-IMPL-072 |

---

## Appendix A: Task Summary

### Total Tasks: 73

| Category | Count | `[CODEX]` | `[ARCHITECT]` |
|----------|-------|-----------|---------------|
| RealTelemetryAdapter | 19 | 18 | 1 |
| DivergencePatternClassifier | 17 | 16 | 1 |
| P5AcceptanceGate | 14 | 13 | 1 |
| TwinCalibration | 15 | 14 | 1 |
| Live Wire Run | 8 | 5 | 3 |
| **Total** | **73** | **66** | **7** |

### Execution Order

```
Phase 1: Foundation (P5-IMPL-001..007, 020..023, 037..040, 051..054)
  - Create all module skeletons and dataclasses
  - Parallelizable across modules

Phase 2: Core Logic (P5-IMPL-008..016, 024..032, 041..046, 055..060)
  - Implement validation, detection, classification, calibration
  - Sequential within module, parallel across modules

Phase 3: Tests (P5-IMPL-017..018, 033, 047, 061..063)
  - Write test suites for all components
  - Requires Phase 2 completion

Phase 4: Integration (P5-IMPL-034..035, 048..049, 064)
  - Connect components to existing infrastructure
  - Requires Phase 3 passing

Phase 5: Live Wire Prep (P5-IMPL-066..070)
  - Build harness and automation
  - Requires Phase 4 completion

Phase 6: [ARCHITECT] Review (P5-IMPL-019, 036, 050, 065, 071..073)
  - Human review of thresholds and execution
  - Gates between phases
```

---

## Appendix B: File Creation Summary

| File Path | Status |
|-----------|--------|
| `backend/topology/first_light/real_telemetry_adapter.py` | TO CREATE |
| `backend/topology/first_light/divergence_patterns.py` | TO CREATE |
| `backend/health/p5_acceptance_gate.py` | TO CREATE |
| `backend/topology/first_light/twin_calibration.py` | TO CREATE |
| `tests/topology/test_telemetry_validator.py` | TO CREATE |
| `tests/topology/test_mock_detector.py` | TO CREATE |
| `tests/topology/test_divergence_patterns.py` | TO CREATE |
| `tests/health/test_p5_acceptance_gate.py` | TO CREATE |
| `tests/topology/test_twin_calibration.py` | TO CREATE |
| `tests/topology/fixtures/calibration_dataset.py` | TO CREATE |
| `scripts/p5_live_wire_harness.py` | TO CREATE |

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-11 | System Law Architect | Initial blueprint |

---

## References

1. [Real_Telemetry_Topology_Spec.md](Real_Telemetry_Topology_Spec.md) — P5 RTTS (upstream spec)
2. [Phase_X_Prelaunch_Review.md](Phase_X_Prelaunch_Review.md) — Go/No-Go criteria
3. [Phase_X_Divergence_Metric.md](Phase_X_Divergence_Metric.md) — Divergence severity bands
4. [Phase_X_P4_Spec.md](Phase_X_P4_Spec.md) — P4 design spec
5. `backend/topology/first_light/runner_p4.py` — Existing P4 implementation
6. `backend/topology/first_light/divergence_analyzer.py` — Existing divergence analysis
