"""
P5 Budget Calibration Harness

This module implements the 3-phase calibration experiment defined in
docs/system_law/Budget_PhaseX_Doctrine.md Section 7.3.

Components:
- SyntheticBudgetTraceGenerator: Phase-1 deterministic budget traces
- CalibrationMeasurementCollector: Phase-2/3 real-load measurement
- BudgetFPFNClassifier: False positive/negative classification
- CalibrationLogEntry: Structured logging per Section 7.3.2

SHADOW MODE CONTRACT:
- All calibration is observational
- No enforcement changes based on calibration results
- Used to validate multiplier coefficients before enablement

See: docs/system_law/Budget_PhaseX_Doctrine.md Section 7.3
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple

from backend.topology.first_light.budget_binding import (
    BudgetDriftClass,
    BudgetStabilityClass,
    build_budget_risk_signal,
    drift_class_from_value,
    stability_class_from_health,
    BudgetRiskSignal,
)

__all__ = [
    # Enums
    "CalibrationPhase",
    "DriftClassificationLabel",
    "BudgetHealthLabel",
    # Data classes
    "CalibrationLogEntry",
    "SyntheticBudgetTrace",
    "CalibrationResult",
    "FPFNReport",
    # Generators
    "SyntheticBudgetTraceGenerator",
    # Collectors
    "CalibrationMeasurementCollector",
    # Classifiers
    "BudgetFPFNClassifier",
    # Harness
    "CalibrationHarness",
]


# =============================================================================
# Enums
# =============================================================================

class CalibrationPhase(str, Enum):
    """Calibration experiment phases."""
    PHASE_1_BASELINE = "PHASE_1_BASELINE"
    PHASE_2_CONTROLLED = "PHASE_2_CONTROLLED"
    PHASE_3_STRESS = "PHASE_3_STRESS"


class DriftClassificationLabel(str, Enum):
    """Ground truth labels for drift classification."""
    NONE = "NONE"
    TRANSIENT = "TRANSIENT"
    PERSISTENT = "PERSISTENT"


class BudgetHealthLabel(str, Enum):
    """Budget health labels for ground truth."""
    SAFE = "SAFE"
    TIGHT = "TIGHT"
    STARVED = "STARVED"
    UNKNOWN = "UNKNOWN"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SyntheticBudgetTrace:
    """
    A single synthetic budget trace for Phase-1 calibration.

    Contains deterministic budget behavior for ground truth establishment.
    """
    cycle: int
    expected_budget: float
    actual_spent: float
    drift_value: float
    health_score: float
    stability_index: float

    # Ground truth labels
    injected_fault: bool = False
    known_stress: bool = False
    expected_drift_class: BudgetDriftClass = BudgetDriftClass.STABLE
    expected_stability_class: BudgetStabilityClass = BudgetStabilityClass.STABLE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "cycle": self.cycle,
            "expected_budget": round(self.expected_budget, 4),
            "actual_spent": round(self.actual_spent, 4),
            "drift_value": round(self.drift_value, 6),
            "health_score": round(self.health_score, 2),
            "stability_index": round(self.stability_index, 4),
            "injected_fault": self.injected_fault,
            "known_stress": self.known_stress,
            "expected_drift_class": self.expected_drift_class.value,
            "expected_stability_class": self.expected_stability_class.value,
        }


@dataclass
class CalibrationLogEntry:
    """
    Calibration log entry per Section 7.3.2.

    Contains all fields required for FP/FN analysis.
    """
    # Core identification
    cycle: int
    timestamp: str
    phase: CalibrationPhase

    # Budget metrics
    expected_budget: float
    actual_spent: float
    drift_value: float
    health_score: float
    stability_index: float

    # Classification (from budget_binding)
    drift_class: BudgetDriftClass
    stability_class: BudgetStabilityClass
    noise_multiplier: float
    severity_multiplier: float
    admissibility_hint: str

    # Ground truth
    manual_label: Optional[str] = None
    injected_fault: bool = False
    known_stress: bool = False

    # Environment
    load_factor: float = 1.0
    gc_occurred: bool = False
    burst_active: bool = False
    warmup_period: bool = False

    # Derived (computed by classifier)
    fp_candidate: bool = False
    fn_candidate: bool = False
    review_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict matching Section 7.3.2 schema."""
        return {
            "calibration_log": {
                "cycle": self.cycle,
                "timestamp": self.timestamp,
                "phase": self.phase.value,

                "budget_metrics": {
                    "expected_budget": round(self.expected_budget, 4),
                    "actual_spent": round(self.actual_spent, 4),
                    "drift_value": round(self.drift_value, 6),
                    "health_score": round(self.health_score, 2),
                    "stability_index": round(self.stability_index, 4),
                },

                "classification": {
                    "drift_class": self.drift_class.value,
                    "stability_class": self.stability_class.value,
                    "noise_multiplier": round(self.noise_multiplier, 4),
                    "severity_multiplier": round(self.severity_multiplier, 4),
                    "admissibility_hint": self.admissibility_hint,
                },

                "ground_truth": {
                    "manual_label": self.manual_label,
                    "injected_fault": self.injected_fault,
                    "known_stress": self.known_stress,
                },

                "environment": {
                    "load_factor": round(self.load_factor, 2),
                    "gc_occurred": self.gc_occurred,
                    "burst_active": self.burst_active,
                    "warmup_period": self.warmup_period,
                },

                "derived": {
                    "fp_candidate": self.fp_candidate,
                    "fn_candidate": self.fn_candidate,
                    "review_required": self.review_required,
                },
            }
        }


@dataclass
class FPFNReport:
    """
    False Positive / False Negative analysis report.
    """
    phase: CalibrationPhase
    total_cycles: int

    # FP analysis
    total_non_stable_classifications: int
    false_positive_count: int
    fp_rate: float

    # FN analysis
    total_actual_stress_cycles: int
    false_negative_count: int
    fn_rate: float

    # Breakdowns
    fp_by_class: Dict[str, int] = field(default_factory=dict)
    fn_by_class: Dict[str, int] = field(default_factory=dict)

    # Validation
    meets_phase_criteria: bool = False
    validation_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "phase": self.phase.value,
            "total_cycles": self.total_cycles,
            "fp_analysis": {
                "total_non_stable_classifications": self.total_non_stable_classifications,
                "false_positive_count": self.false_positive_count,
                "fp_rate": round(self.fp_rate, 4),
                "fp_by_class": self.fp_by_class,
            },
            "fn_analysis": {
                "total_actual_stress_cycles": self.total_actual_stress_cycles,
                "false_negative_count": self.false_negative_count,
                "fn_rate": round(self.fn_rate, 4),
                "fn_by_class": self.fn_by_class,
            },
            "meets_phase_criteria": self.meets_phase_criteria,
            "validation_notes": self.validation_notes,
        }


@dataclass
class CalibrationResult:
    """
    Complete calibration experiment result.
    """
    experiment_id: str
    start_time: str
    end_time: str

    phase_1_report: Optional[FPFNReport] = None
    phase_2_report: Optional[FPFNReport] = None
    phase_3_report: Optional[FPFNReport] = None

    overall_pass: bool = False
    enablement_recommendation: str = "NOT_RECOMMENDED"

    coefficient_adjustments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "experiment_id": self.experiment_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "phase_1_report": self.phase_1_report.to_dict() if self.phase_1_report else None,
            "phase_2_report": self.phase_2_report.to_dict() if self.phase_2_report else None,
            "phase_3_report": self.phase_3_report.to_dict() if self.phase_3_report else None,
            "overall_pass": self.overall_pass,
            "enablement_recommendation": self.enablement_recommendation,
            "coefficient_adjustments": self.coefficient_adjustments,
        }


# =============================================================================
# Phase-1: Synthetic Baseline Generator
# =============================================================================

class SyntheticBudgetTraceGenerator:
    """
    Generates deterministic synthetic budget traces for Phase-1 calibration.

    Features:
    - Deterministic PRNG for reproducibility
    - Configurable fault injection (10 known budget exhaustions)
    - Controlled drift distribution: 5 STABLE, 3 DRIFTING, 2 VOLATILE windows

    Per Section 7.3.1 Phase 1:
    - Duration: 500 cycles
    - Load profile: Synthetic, deterministic
    - Injected faults: 10 known budget exhaustions
    - Expected drift: 5 STABLE, 3 DRIFTING, 2 VOLATILE windows
    """

    # Default configuration per doctrine
    DEFAULT_CYCLES = 500
    DEFAULT_EXPECTED_BUDGET = 100.0
    DEFAULT_FAULT_COUNT = 10

    # Window distribution for 500 cycles (50 windows of 10 cycles each)
    # 5 STABLE windows, 3 DRIFTING windows, 2 VOLATILE windows per 50-window set
    WINDOW_SIZE = 10

    def __init__(
        self,
        seed: int = 42,
        cycles: int = DEFAULT_CYCLES,
        expected_budget: float = DEFAULT_EXPECTED_BUDGET,
        fault_count: int = DEFAULT_FAULT_COUNT,
    ):
        """
        Initialize generator with deterministic seed.

        Args:
            seed: Random seed for reproducibility
            cycles: Total number of cycles to generate
            expected_budget: Nominal budget per cycle
            fault_count: Number of fault injections
        """
        self.seed = seed
        self.cycles = cycles
        self.expected_budget = expected_budget
        self.fault_count = fault_count

        # Deterministic pseudo-random state
        self._state = seed

        # Pre-compute fault injection cycles
        self._fault_cycles = self._compute_fault_cycles()

        # Pre-compute window drift profiles
        self._window_profiles = self._compute_window_profiles()

    def _deterministic_random(self) -> float:
        """
        Generate deterministic pseudo-random float in [0, 1).

        Uses simple LCG for reproducibility.
        """
        # LCG parameters (same as MINSTD)
        a = 48271
        m = 2147483647
        self._state = (a * self._state) % m
        return self._state / m

    def _compute_fault_cycles(self) -> List[int]:
        """
        Pre-compute cycles where faults will be injected.

        Distributes 10 faults evenly across 500 cycles with jitter.
        """
        faults = []
        spacing = self.cycles // self.fault_count

        for i in range(self.fault_count):
            base = i * spacing + spacing // 2
            # Add small deterministic jitter
            jitter = int((self._deterministic_random() - 0.5) * spacing * 0.3)
            cycle = max(0, min(self.cycles - 1, base + jitter))
            faults.append(cycle)

        return sorted(faults)

    def _compute_window_profiles(self) -> Dict[int, Tuple[str, float, float]]:
        """
        Pre-compute drift profile for each window.

        Returns dict mapping window_index to (drift_class, drift_value, health_score).

        Distribution per 50-window set:
        - 5 windows: STABLE (drift ~0.02, health ~85)
        - 3 windows: DRIFTING (drift ~0.10, health ~75)
        - 2 windows: VOLATILE (drift ~0.20, health ~65)
        """
        profiles = {}
        num_windows = self.cycles // self.WINDOW_SIZE

        for w in range(num_windows):
            # Pattern repeats every 10 windows: 5 STABLE, 3 DRIFTING, 2 VOLATILE
            pattern_pos = w % 10

            if pattern_pos < 5:
                # STABLE windows (0-4)
                drift = 0.02 + self._deterministic_random() * 0.02  # [0.02, 0.04]
                health = 82.0 + self._deterministic_random() * 8.0  # [82, 90]
                stability = 0.96 + self._deterministic_random() * 0.03  # [0.96, 0.99]
                profiles[w] = ("STABLE", drift, health, stability)
            elif pattern_pos < 8:
                # DRIFTING windows (5-7)
                drift = 0.08 + self._deterministic_random() * 0.05  # [0.08, 0.13]
                health = 72.0 + self._deterministic_random() * 6.0  # [72, 78]
                stability = 0.80 + self._deterministic_random() * 0.10  # [0.80, 0.90]
                profiles[w] = ("DRIFTING", drift, health, stability)
            else:
                # VOLATILE windows (8-9)
                drift = 0.18 + self._deterministic_random() * 0.08  # [0.18, 0.26]
                health = 62.0 + self._deterministic_random() * 6.0  # [62, 68]
                stability = 0.60 + self._deterministic_random() * 0.08  # [0.60, 0.68]
                profiles[w] = ("VOLATILE", drift, health, stability)

        return profiles

    def generate(self) -> Iterator[SyntheticBudgetTrace]:
        """
        Generate synthetic budget traces.

        Yields:
            SyntheticBudgetTrace for each cycle
        """
        # Reset state for reproducibility
        self._state = self.seed
        self._fault_cycles = self._compute_fault_cycles()
        self._window_profiles = self._compute_window_profiles()

        for cycle in range(self.cycles):
            window_idx = cycle // self.WINDOW_SIZE
            profile = self._window_profiles.get(window_idx, ("STABLE", 0.02, 85.0, 0.97))

            class_name, base_drift, base_health, base_stability = profile

            # Add per-cycle variation
            drift_var = (self._deterministic_random() - 0.5) * 0.01
            health_var = (self._deterministic_random() - 0.5) * 2.0
            stability_var = (self._deterministic_random() - 0.5) * 0.02

            drift_value = base_drift + drift_var
            health_score = base_health + health_var
            stability_index = base_stability + stability_var

            # Check for fault injection
            injected_fault = cycle in self._fault_cycles
            if injected_fault:
                # Fault injection: severe budget exhaustion
                drift_value = 0.30 + self._deterministic_random() * 0.10  # [0.30, 0.40]
                health_score = 45.0 + self._deterministic_random() * 10.0  # [45, 55]
                stability_index = 0.50 + self._deterministic_random() * 0.10  # [0.50, 0.60]

            # Compute actual_spent from drift
            actual_spent = self.expected_budget * (1.0 + drift_value)

            # Determine expected classifications
            expected_drift_class = drift_class_from_value(drift_value)
            expected_stability_class = stability_class_from_health(health_score, stability_index)

            # Determine known_stress
            known_stress = (
                expected_drift_class != BudgetDriftClass.STABLE or
                expected_stability_class != BudgetStabilityClass.STABLE or
                injected_fault
            )

            yield SyntheticBudgetTrace(
                cycle=cycle,
                expected_budget=self.expected_budget,
                actual_spent=actual_spent,
                drift_value=drift_value,
                health_score=health_score,
                stability_index=stability_index,
                injected_fault=injected_fault,
                known_stress=known_stress,
                expected_drift_class=expected_drift_class,
                expected_stability_class=expected_stability_class,
            )

    def generate_all(self) -> List[SyntheticBudgetTrace]:
        """Generate all traces as a list."""
        return list(self.generate())


# =============================================================================
# Phase-2/3: Measurement Collector
# =============================================================================

class CalibrationMeasurementCollector:
    """
    Collects calibration measurements for Phase-2/3.

    Features:
    - Collects budget metrics from real or simulated load
    - Generates CalibrationLogEntry with all required fields
    - Tracks environment context (load_factor, gc, burst, warmup)
    """

    def __init__(self, phase: CalibrationPhase):
        """
        Initialize collector for a specific phase.

        Args:
            phase: Calibration phase (PHASE_2_CONTROLLED or PHASE_3_STRESS)
        """
        self.phase = phase
        self.entries: List[CalibrationLogEntry] = []
        self.cycle_counter = 0

    def collect(
        self,
        expected_budget: float,
        actual_spent: float,
        health_score: float,
        stability_index: float,
        # Optional environment context
        load_factor: float = 1.0,
        gc_occurred: bool = False,
        burst_active: bool = False,
        warmup_period: bool = False,
        # Optional ground truth
        manual_label: Optional[str] = None,
        injected_fault: bool = False,
        known_stress: bool = False,
    ) -> CalibrationLogEntry:
        """
        Collect a single measurement.

        Args:
            expected_budget: Nominal budget allocation
            actual_spent: Actual budget consumed
            health_score: Budget health score (0-100)
            stability_index: Stability index (0.0-1.0)
            load_factor: Current load factor (1.0 = nominal)
            gc_occurred: Whether GC occurred this cycle
            burst_active: Whether burst is active
            warmup_period: Whether in warmup period
            manual_label: Manual ground truth label
            injected_fault: Whether fault was injected
            known_stress: Whether stress is known

        Returns:
            CalibrationLogEntry with all fields populated
        """
        # Compute drift
        drift_value = (actual_spent - expected_budget) / expected_budget if expected_budget > 0 else 0.0

        # Build risk signal for classification
        signal = build_budget_risk_signal(
            drift_value=drift_value,
            health_score=health_score,
            stability_index=stability_index,
        )

        # Create entry
        entry = CalibrationLogEntry(
            cycle=self.cycle_counter,
            timestamp=datetime.now(timezone.utc).isoformat(),
            phase=self.phase,
            expected_budget=expected_budget,
            actual_spent=actual_spent,
            drift_value=drift_value,
            health_score=health_score,
            stability_index=stability_index,
            drift_class=signal.drift_class,
            stability_class=signal.stability_class,
            noise_multiplier=signal.noise_multiplier,
            severity_multiplier=signal.severity_multiplier,
            admissibility_hint=signal.admissibility_hint,
            manual_label=manual_label,
            injected_fault=injected_fault,
            known_stress=known_stress,
            load_factor=load_factor,
            gc_occurred=gc_occurred,
            burst_active=burst_active,
            warmup_period=warmup_period,
        )

        self.entries.append(entry)
        self.cycle_counter += 1

        return entry

    def collect_from_trace(self, trace: SyntheticBudgetTrace) -> CalibrationLogEntry:
        """
        Collect measurement from a synthetic trace.

        Convenience method for Phase-1 integration.

        Args:
            trace: SyntheticBudgetTrace to convert

        Returns:
            CalibrationLogEntry
        """
        return self.collect(
            expected_budget=trace.expected_budget,
            actual_spent=trace.actual_spent,
            health_score=trace.health_score,
            stability_index=trace.stability_index,
            injected_fault=trace.injected_fault,
            known_stress=trace.known_stress,
        )

    def get_entries(self) -> List[CalibrationLogEntry]:
        """Get all collected entries."""
        return list(self.entries)

    def clear(self):
        """Clear collected entries and reset counter."""
        self.entries.clear()
        self.cycle_counter = 0


# =============================================================================
# FP/FN Classifier
# =============================================================================

class BudgetFPFNClassifier:
    """
    Classifies False Positives and False Negatives per Section 7.3.2.

    FP Definition: Budget classified as DRIFTING/VOLATILE when actually STABLE.
    - actual_spent within 5% of expected AND health >= 80 AND stability >= 0.95
    - Classification should be STABLE
    - Any non-STABLE classification = FP

    FN Definition: Budget classified as STABLE when actually DRIFTING/VOLATILE.
    - actual_spent > 1.15 × expected (15% over-utilization)
    - OR health < 70 OR stability < 0.7
    - OR any INV-BUD-* violation
    - STABLE classification during these conditions = FN
    """

    # Thresholds from doctrine
    FP_DRIFT_THRESHOLD = 0.05  # 5% drift tolerance for "actually STABLE"
    FP_HEALTH_THRESHOLD = 80.0
    FP_STABILITY_THRESHOLD = 0.95

    FN_OVERUTIL_THRESHOLD = 0.15  # 15% over-utilization
    FN_HEALTH_THRESHOLD = 70.0
    FN_STABILITY_THRESHOLD = 0.7

    def __init__(self):
        """Initialize classifier."""
        pass

    def is_actually_stable(self, entry: CalibrationLogEntry) -> bool:
        """
        Determine if budget is actually STABLE (ground truth).

        Criteria (all must be true):
        - drift_value within ±5% of expected
        - health_score >= 80
        - stability_index >= 0.95
        - Not during known_stress or injected_fault

        Args:
            entry: CalibrationLogEntry to evaluate

        Returns:
            True if budget is actually stable
        """
        if entry.injected_fault or entry.known_stress:
            return False

        drift_within_tolerance = abs(entry.drift_value) <= self.FP_DRIFT_THRESHOLD
        health_ok = entry.health_score >= self.FP_HEALTH_THRESHOLD
        stability_ok = entry.stability_index >= self.FP_STABILITY_THRESHOLD

        return drift_within_tolerance and health_ok and stability_ok

    def is_actually_stressed(self, entry: CalibrationLogEntry) -> bool:
        """
        Determine if budget is actually STRESSED (ground truth).

        Criteria (any is true):
        - actual_spent > 1.15 × expected (15% over-utilization)
        - health_score < 70
        - stability_index < 0.7
        - injected_fault = True

        Args:
            entry: CalibrationLogEntry to evaluate

        Returns:
            True if budget is actually stressed
        """
        if entry.injected_fault:
            return True

        over_utilized = entry.drift_value > self.FN_OVERUTIL_THRESHOLD
        health_low = entry.health_score < self.FN_HEALTH_THRESHOLD
        stability_low = entry.stability_index < self.FN_STABILITY_THRESHOLD

        return over_utilized or health_low or stability_low

    def classify_entry(self, entry: CalibrationLogEntry) -> CalibrationLogEntry:
        """
        Classify an entry for FP/FN and set derived fields.

        Modifies entry in-place and returns it.

        Args:
            entry: CalibrationLogEntry to classify

        Returns:
            Same entry with fp_candidate, fn_candidate, review_required set
        """
        actually_stable = self.is_actually_stable(entry)
        actually_stressed = self.is_actually_stressed(entry)

        classified_stable = entry.drift_class == BudgetDriftClass.STABLE
        classified_non_stable = entry.drift_class != BudgetDriftClass.STABLE

        # FP: Classified as non-STABLE when actually STABLE
        entry.fp_candidate = classified_non_stable and actually_stable

        # FN: Classified as STABLE when actually STRESSED
        entry.fn_candidate = classified_stable and actually_stressed

        # Review required if either FP or FN candidate
        entry.review_required = entry.fp_candidate or entry.fn_candidate

        return entry

    def classify_all(self, entries: List[CalibrationLogEntry]) -> List[CalibrationLogEntry]:
        """
        Classify all entries.

        Args:
            entries: List of entries to classify

        Returns:
            Same entries with derived fields set
        """
        return [self.classify_entry(e) for e in entries]

    def generate_report(
        self,
        entries: List[CalibrationLogEntry],
        phase: CalibrationPhase,
    ) -> FPFNReport:
        """
        Generate FP/FN report for a phase.

        Args:
            entries: Classified entries
            phase: Calibration phase

        Returns:
            FPFNReport with analysis results
        """
        total_cycles = len(entries)

        # FP analysis
        non_stable_classifications = [e for e in entries if e.drift_class != BudgetDriftClass.STABLE]
        fp_entries = [e for e in entries if e.fp_candidate]

        total_non_stable = len(non_stable_classifications)
        fp_count = len(fp_entries)
        fp_rate = fp_count / total_non_stable if total_non_stable > 0 else 0.0

        # FP breakdown by class
        fp_by_class = {}
        for e in fp_entries:
            cls = e.drift_class.value
            fp_by_class[cls] = fp_by_class.get(cls, 0) + 1

        # FN analysis
        stressed_entries = [e for e in entries if self.is_actually_stressed(e)]
        fn_entries = [e for e in entries if e.fn_candidate]

        total_stressed = len(stressed_entries)
        fn_count = len(fn_entries)
        fn_rate = fn_count / total_stressed if total_stressed > 0 else 0.0

        # FN breakdown by expected class
        fn_by_class = {}
        for e in fn_entries:
            # Use stability class since drift class is STABLE (by definition of FN)
            cls = e.stability_class.value
            fn_by_class[cls] = fn_by_class.get(cls, 0) + 1

        # Determine if phase criteria met
        validation_notes = []

        if phase == CalibrationPhase.PHASE_1_BASELINE:
            # Phase 1: Classification accuracy >= 99%
            accuracy = (total_cycles - fp_count - fn_count) / total_cycles if total_cycles > 0 else 0.0
            meets_criteria = accuracy >= 0.99
            validation_notes.append(f"Classification accuracy: {accuracy:.2%}")
            if not meets_criteria:
                validation_notes.append("FAIL: Accuracy below 99% threshold")

        elif phase == CalibrationPhase.PHASE_2_CONTROLLED:
            # Phase 2: FP <= 5%, FN <= 2%
            fp_ok = fp_rate <= 0.05
            fn_ok = fn_rate <= 0.02
            meets_criteria = fp_ok and fn_ok

            if not fp_ok:
                validation_notes.append(f"FAIL: FP rate {fp_rate:.2%} exceeds 5% threshold")
            if not fn_ok:
                validation_notes.append(f"FAIL: FN rate {fn_rate:.2%} exceeds 2% threshold")
            if meets_criteria:
                validation_notes.append("Phase 2 criteria met")

        else:  # PHASE_3_STRESS
            # Phase 3: FP <= 10%, FN <= 5%
            fp_ok = fp_rate <= 0.10
            fn_ok = fn_rate <= 0.05
            meets_criteria = fp_ok and fn_ok

            if not fp_ok:
                validation_notes.append(f"FAIL: FP rate {fp_rate:.2%} exceeds 10% threshold")
            if not fn_ok:
                validation_notes.append(f"FAIL: FN rate {fn_rate:.2%} exceeds 5% threshold")
            if meets_criteria:
                validation_notes.append("Phase 3 criteria met")

        return FPFNReport(
            phase=phase,
            total_cycles=total_cycles,
            total_non_stable_classifications=total_non_stable,
            false_positive_count=fp_count,
            fp_rate=fp_rate,
            total_actual_stress_cycles=total_stressed,
            false_negative_count=fn_count,
            fn_rate=fn_rate,
            fp_by_class=fp_by_class,
            fn_by_class=fn_by_class,
            meets_phase_criteria=meets_criteria,
            validation_notes=validation_notes,
        )


# =============================================================================
# Calibration Harness
# =============================================================================

class CalibrationHarness:
    """
    3-Phase Calibration Harness orchestrating the full experiment.

    Integrates:
    - Phase-1: Synthetic baseline (500 cycles)
    - Phase-2: Controlled real load (1000 cycles)
    - Phase-3: Stress load (500 cycles)

    Per Section 7.3.1.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize harness.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.classifier = BudgetFPFNClassifier()

        self.phase_1_entries: List[CalibrationLogEntry] = []
        self.phase_2_entries: List[CalibrationLogEntry] = []
        self.phase_3_entries: List[CalibrationLogEntry] = []

        self.result: Optional[CalibrationResult] = None

    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now(timezone.utc).isoformat()
        data = f"calibration-{timestamp}-{self.seed}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def run_phase_1(self, cycles: int = 500) -> FPFNReport:
        """
        Run Phase-1: Synthetic Baseline.

        Args:
            cycles: Number of cycles (default 500)

        Returns:
            FPFNReport for Phase-1
        """
        generator = SyntheticBudgetTraceGenerator(seed=self.seed, cycles=cycles)
        collector = CalibrationMeasurementCollector(CalibrationPhase.PHASE_1_BASELINE)

        for trace in generator.generate():
            collector.collect_from_trace(trace)

        self.phase_1_entries = collector.get_entries()
        self.classifier.classify_all(self.phase_1_entries)

        return self.classifier.generate_report(
            self.phase_1_entries,
            CalibrationPhase.PHASE_1_BASELINE,
        )

    def run_phase_2(
        self,
        measurements: List[Dict[str, Any]],
    ) -> FPFNReport:
        """
        Run Phase-2: Controlled Real Load.

        Args:
            measurements: List of measurement dicts with keys:
                - expected_budget
                - actual_spent
                - health_score
                - stability_index
                - (optional) load_factor, gc_occurred, burst_active, warmup_period
                - (optional) manual_label, known_stress

        Returns:
            FPFNReport for Phase-2
        """
        collector = CalibrationMeasurementCollector(CalibrationPhase.PHASE_2_CONTROLLED)

        for m in measurements:
            collector.collect(
                expected_budget=m["expected_budget"],
                actual_spent=m["actual_spent"],
                health_score=m["health_score"],
                stability_index=m["stability_index"],
                load_factor=m.get("load_factor", 1.0),
                gc_occurred=m.get("gc_occurred", False),
                burst_active=m.get("burst_active", False),
                warmup_period=m.get("warmup_period", False),
                manual_label=m.get("manual_label"),
                known_stress=m.get("known_stress", False),
            )

        self.phase_2_entries = collector.get_entries()
        self.classifier.classify_all(self.phase_2_entries)

        return self.classifier.generate_report(
            self.phase_2_entries,
            CalibrationPhase.PHASE_2_CONTROLLED,
        )

    def run_phase_3(
        self,
        measurements: List[Dict[str, Any]],
    ) -> FPFNReport:
        """
        Run Phase-3: Stress Load.

        Args:
            measurements: List of measurement dicts (same format as Phase-2)

        Returns:
            FPFNReport for Phase-3
        """
        collector = CalibrationMeasurementCollector(CalibrationPhase.PHASE_3_STRESS)

        for m in measurements:
            collector.collect(
                expected_budget=m["expected_budget"],
                actual_spent=m["actual_spent"],
                health_score=m["health_score"],
                stability_index=m["stability_index"],
                load_factor=m.get("load_factor", 2.0),  # Default to 2x for stress
                gc_occurred=m.get("gc_occurred", False),
                burst_active=m.get("burst_active", False),
                warmup_period=m.get("warmup_period", False),
                manual_label=m.get("manual_label"),
                injected_fault=m.get("injected_fault", False),
                known_stress=m.get("known_stress", False),
            )

        self.phase_3_entries = collector.get_entries()
        self.classifier.classify_all(self.phase_3_entries)

        return self.classifier.generate_report(
            self.phase_3_entries,
            CalibrationPhase.PHASE_3_STRESS,
        )

    def generate_synthetic_phase_2_data(
        self,
        cycles: int = 1000,
        seed: int = 123,
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic Phase-2 data for testing.

        This simulates production-like behavior with ±20% variance
        from synthetic baseline, as specified in Section 7.3.1.

        Args:
            cycles: Number of cycles
            seed: Random seed

        Returns:
            List of measurement dicts
        """
        measurements = []
        state = seed

        def rand():
            nonlocal state
            state = (48271 * state) % 2147483647
            return state / 2147483647

        for i in range(cycles):
            # Base values with production-like variation
            drift_base = 0.03 + (rand() - 0.5) * 0.08  # [−0.01, 0.07] mostly STABLE
            health_base = 80.0 + (rand() - 0.5) * 15.0  # [72.5, 87.5]
            stability_base = 0.92 + (rand() - 0.5) * 0.10  # [0.87, 0.97]

            # Occasional stress injection (5% of cycles)
            if rand() < 0.05:
                drift_base = 0.12 + rand() * 0.08  # [0.12, 0.20]
                health_base = 65.0 + rand() * 10.0  # [65, 75]
                stability_base = 0.70 + rand() * 0.15  # [0.70, 0.85]

            expected = 100.0
            actual = expected * (1.0 + drift_base)

            measurements.append({
                "expected_budget": expected,
                "actual_spent": actual,
                "health_score": health_base,
                "stability_index": stability_base,
                "load_factor": 1.0,
                "gc_occurred": rand() < 0.02,  # 2% GC rate
                "burst_active": False,
                "warmup_period": i < 20,
                "known_stress": drift_base > 0.10 or health_base < 70,
            })

        return measurements

    def generate_synthetic_phase_3_data(
        self,
        cycles: int = 500,
        seed: int = 456,
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic Phase-3 stress data for testing.

        Simulates 2× load with burst injection every 20 cycles
        and GC every 100 cycles.

        Args:
            cycles: Number of cycles
            seed: Random seed

        Returns:
            List of measurement dicts
        """
        measurements = []
        state = seed

        def rand():
            nonlocal state
            state = (48271 * state) % 2147483647
            return state / 2147483647

        for i in range(cycles):
            # 2× load baseline with higher variance
            drift_base = 0.08 + (rand() - 0.5) * 0.15  # [0.005, 0.155]
            health_base = 72.0 + (rand() - 0.5) * 20.0  # [62, 82]
            stability_base = 0.80 + (rand() - 0.5) * 0.20  # [0.70, 0.90]

            burst_active = (i % 20 < 3)  # 50ms burst every 20 cycles
            gc_occurred = (i % 100 == 0)  # GC every 100 cycles

            if burst_active:
                drift_base += 0.10  # Burst adds 10% drift
                health_base -= 10.0

            if gc_occurred:
                stability_base -= 0.15  # GC reduces stability

            expected = 100.0
            actual = expected * (1.0 + drift_base)

            measurements.append({
                "expected_budget": expected,
                "actual_spent": actual,
                "health_score": max(40.0, health_base),
                "stability_index": max(0.4, stability_base),
                "load_factor": 2.0,
                "gc_occurred": gc_occurred,
                "burst_active": burst_active,
                "warmup_period": False,
                "known_stress": drift_base > 0.10 or health_base < 70 or burst_active,
            })

        return measurements

    def run_full_experiment(self) -> CalibrationResult:
        """
        Run complete 3-phase calibration experiment.

        Uses synthetic data for all phases (for testing).
        For production, replace Phase 2/3 with real measurements.

        Returns:
            CalibrationResult with all phase reports
        """
        experiment_id = self._generate_experiment_id()
        start_time = datetime.now(timezone.utc).isoformat()

        # Phase 1: Synthetic baseline
        phase_1_report = self.run_phase_1(cycles=500)

        # Phase 2: Controlled (synthetic for testing)
        phase_2_data = self.generate_synthetic_phase_2_data(cycles=1000)
        phase_2_report = self.run_phase_2(phase_2_data)

        # Phase 3: Stress (synthetic for testing)
        phase_3_data = self.generate_synthetic_phase_3_data(cycles=500)
        phase_3_report = self.run_phase_3(phase_3_data)

        end_time = datetime.now(timezone.utc).isoformat()

        # Determine overall pass
        overall_pass = (
            phase_1_report.meets_phase_criteria and
            phase_2_report.meets_phase_criteria and
            phase_3_report.meets_phase_criteria
        )

        # Generate enablement recommendation
        if overall_pass:
            enablement_recommendation = "PROCEED_TO_STAGE_2"
        elif phase_1_report.meets_phase_criteria and phase_2_report.meets_phase_criteria:
            enablement_recommendation = "PHASE_3_REMEDIATION_NEEDED"
        elif phase_1_report.meets_phase_criteria:
            enablement_recommendation = "PHASE_2_REMEDIATION_NEEDED"
        else:
            enablement_recommendation = "PHASE_1_BASELINE_FAILED"

        self.result = CalibrationResult(
            experiment_id=experiment_id,
            start_time=start_time,
            end_time=end_time,
            phase_1_report=phase_1_report,
            phase_2_report=phase_2_report,
            phase_3_report=phase_3_report,
            overall_pass=overall_pass,
            enablement_recommendation=enablement_recommendation,
        )

        return self.result

    def get_all_entries(self) -> List[CalibrationLogEntry]:
        """Get all entries from all phases."""
        return self.phase_1_entries + self.phase_2_entries + self.phase_3_entries

    def export_logs_json(self) -> str:
        """Export all calibration logs as JSON."""
        all_logs = [e.to_dict() for e in self.get_all_entries()]
        return json.dumps(all_logs, indent=2)
