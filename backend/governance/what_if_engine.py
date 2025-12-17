"""
What-If Governance Engine — Hypothetical Gate Analysis for Phase Y.

SHADOW MODE ONLY: This engine evaluates what gates WOULD have done,
but takes NO enforcement action. All output is observational.

Minimal prototype using only:
- G2: Invariant inputs
- G3: Omega state
- G4: RSI streaks

Usage:
    from backend.governance.what_if_engine import (
        WhatIfEngine,
        WhatIfCycleInput,
        build_what_if_report,
    )

    engine = WhatIfEngine()

    # Feed cycle data
    for cycle_data in telemetry:
        input = WhatIfCycleInput.from_telemetry(cycle_data)
        result = engine.evaluate_cycle(input)

    # Generate report
    report = engine.build_report()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

__all__ = [
    "WhatIfEngine",
    "WhatIfCycleInput",
    "WhatIfCycleResult",
    "WhatIfReport",
    "GateWhatIfAnalysis",
    "NotableEvent",
    "CalibrationRecommendation",
    "build_what_if_report",
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class WhatIfCycleInput:
    """
    Minimal input for What-If evaluation.

    Only G2, G3, G4 relevant fields.
    """
    cycle: int
    timestamp: str

    # G2: Invariant inputs
    invariant_violations: List[str] = field(default_factory=list)

    # G3: Omega state
    in_omega: bool = True
    omega_exit_streak: int = 0

    # G4: RSI streaks
    rho: float = 1.0
    rho_collapse_streak: int = 0

    # Optional context
    context: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_telemetry(cls, data: Dict[str, Any], cycle: int) -> "WhatIfCycleInput":
        """Build from telemetry dictionary."""
        return cls(
            cycle=cycle,
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            invariant_violations=data.get("invariant_violations", []),
            in_omega=data.get("in_omega", True),
            omega_exit_streak=data.get("omega_exit_streak", 0),
            rho=data.get("rho", data.get("rsi", 1.0)),
            rho_collapse_streak=data.get("rho_collapse_streak", 0),
            context=data.get("context", {}),
        )

    @classmethod
    def from_usla_state(cls, state: Any, cycle: int) -> "WhatIfCycleInput":
        """Build from USLAState object."""
        return cls(
            cycle=cycle,
            timestamp=datetime.now(timezone.utc).isoformat(),
            invariant_violations=getattr(state, "invariant_violations", []),
            in_omega=getattr(state, "is_in_safe_region", lambda _: True)(None) if callable(getattr(state, "is_in_safe_region", None)) else True,
            omega_exit_streak=0,  # Must be tracked externally
            rho=getattr(state, "rho", 1.0),
            rho_collapse_streak=0,  # Must be tracked externally
        )


@dataclass
class WhatIfCycleResult:
    """Result of single cycle What-If evaluation."""
    cycle: int
    timestamp: str

    # Hypothetical verdict
    verdict: Literal["ALLOW", "BLOCK"]
    blocking_gate: Optional[str] = None
    blocking_reason: Optional[str] = None

    # Individual gate results
    g2_status: Literal["PASS", "FAIL"] = "PASS"
    g2_trigger: int = 0  # violation count

    g3_status: Literal["PASS", "FAIL"] = "PASS"
    g3_trigger: int = 0  # omega exit streak

    g4_status: Literal["PASS", "FAIL"] = "PASS"
    g4_trigger: float = 0.0  # rho value
    g4_streak: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle": self.cycle,
            "timestamp": self.timestamp,
            "verdict": self.verdict,
            "blocking_gate": self.blocking_gate,
            "blocking_reason": self.blocking_reason,
            "gates": {
                "g2_invariant": {
                    "status": self.g2_status,
                    "trigger_value": self.g2_trigger,
                },
                "g3_safe_region": {
                    "status": self.g3_status,
                    "trigger_value": self.g3_trigger,
                },
                "g4_soft": {
                    "status": self.g4_status,
                    "trigger_value": self.g4_trigger,
                    "streak": self.g4_streak,
                },
            },
        }


@dataclass
class ThresholdBreach:
    """Record of a threshold breach event."""
    cycle: int
    trigger_value: Any
    duration: int = 1


@dataclass
class GateWhatIfAnalysis:
    """What-If analysis for a single gate."""
    gate_id: str
    hypothetical_fail_count: int = 0
    hypothetical_block_count: int = 0
    fail_rate: float = 0.0
    peak_trigger_value: Any = None
    peak_cycle: Optional[int] = None
    mean_margin_to_threshold: float = 0.0
    threshold_breaches: List[ThresholdBreach] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothetical_fail_count": self.hypothetical_fail_count,
            "hypothetical_block_count": self.hypothetical_block_count,
            "fail_rate": round(self.fail_rate, 4),
            "peak_trigger_value": self.peak_trigger_value,
            "peak_cycle": self.peak_cycle,
            "mean_margin_to_threshold": round(self.mean_margin_to_threshold, 4),
            "threshold_breaches": [
                {"cycle": b.cycle, "trigger_value": b.trigger_value, "duration": b.duration}
                for b in self.threshold_breaches
            ],
        }


@dataclass
class NotableEvent:
    """Notable event during What-If analysis."""
    cycle: int
    event_type: str
    gate_id: str
    description: str
    hypothetical_verdict: Literal["ALLOW", "BLOCK"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle": self.cycle,
            "event_type": self.event_type,
            "gate_id": self.gate_id,
            "description": self.description,
            "hypothetical_verdict": self.hypothetical_verdict,
        }


@dataclass
class CalibrationRecommendation:
    """Threshold calibration recommendation."""
    gate_id: str
    parameter: str
    current_value: Any
    suggested_value: Any
    rationale: str
    confidence: Literal["LOW", "MEDIUM", "HIGH"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "parameter": self.parameter,
            "current_value": self.current_value,
            "suggested_value": self.suggested_value,
            "rationale": self.rationale,
            "confidence": self.confidence,
        }


@dataclass
class WhatIfReport:
    """Complete What-If analysis report."""
    schema_version: str = "1.0.0"
    run_id: str = ""
    analysis_timestamp: str = ""
    mode: str = "HYPOTHETICAL"

    # Summary
    total_cycles: int = 0
    hypothetical_allows: int = 0
    hypothetical_blocks: int = 0
    hypothetical_block_rate: float = 0.0
    blocking_gate_distribution: Dict[str, int] = field(default_factory=dict)
    max_consecutive_blocks: int = 0
    first_hypothetical_block_cycle: Optional[int] = None

    # Gate analysis
    g2_analysis: Optional[GateWhatIfAnalysis] = None
    g3_analysis: Optional[GateWhatIfAnalysis] = None
    g4_analysis: Optional[GateWhatIfAnalysis] = None

    # Events and recommendations
    notable_events: List[NotableEvent] = field(default_factory=list)
    calibration_recommendations: List[CalibrationRecommendation] = field(default_factory=list)

    auditor_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "analysis_timestamp": self.analysis_timestamp,
            "mode": self.mode,
            "summary": {
                "total_cycles": self.total_cycles,
                "hypothetical_allows": self.hypothetical_allows,
                "hypothetical_blocks": self.hypothetical_blocks,
                "hypothetical_block_rate": round(self.hypothetical_block_rate, 4),
                "blocking_gate_distribution": self.blocking_gate_distribution,
                "max_consecutive_blocks": self.max_consecutive_blocks,
                "first_hypothetical_block_cycle": self.first_hypothetical_block_cycle,
            },
            "gate_analysis": {
                "g2_invariant": self.g2_analysis.to_dict() if self.g2_analysis else None,
                "g3_safe_region": self.g3_analysis.to_dict() if self.g3_analysis else None,
                "g4_soft": self.g4_analysis.to_dict() if self.g4_analysis else None,
            },
            "notable_events": [e.to_dict() for e in self.notable_events],
            "calibration_recommendations": [r.to_dict() for r in self.calibration_recommendations],
            "auditor_notes": self.auditor_notes,
        }


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class WhatIfConfig:
    """Configuration for What-If engine thresholds."""
    # G2: Invariant tolerance
    invariant_tolerance: int = 0

    # G3: Omega exit threshold
    omega_exit_threshold: int = 100

    # G4: RSI thresholds
    rho_min: float = 0.4
    rho_streak_threshold: int = 10

    @classmethod
    def default(cls) -> "WhatIfConfig":
        return cls()


# =============================================================================
# WHAT-IF ENGINE
# =============================================================================

class WhatIfEngine:
    """
    Hypothetical Governance Evaluator.

    SHADOW MODE ONLY: Evaluates what gates WOULD have done.
    Takes NO enforcement action.
    """

    def __init__(
        self,
        config: Optional[WhatIfConfig] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self.config = config or WhatIfConfig.default()
        self.run_id = run_id or f"what-if-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

        # Results storage
        self._results: List[WhatIfCycleResult] = []
        self._blocks: List[WhatIfCycleResult] = []

        # Breach tracking for analysis
        self._g2_breach_start: Optional[int] = None
        self._g3_breach_start: Optional[int] = None
        self._g4_breach_start: Optional[int] = None

        self._g2_breaches: List[ThresholdBreach] = []
        self._g3_breaches: List[ThresholdBreach] = []
        self._g4_breaches: List[ThresholdBreach] = []

        # Peak tracking
        self._g2_peak: Tuple[int, int] = (0, 0)  # (value, cycle)
        self._g3_peak: Tuple[int, int] = (0, 0)
        self._g4_peak: Tuple[float, int, int] = (1.0, 0, 0)  # (rho, streak, cycle)

        # Margin tracking for mean computation
        self._g2_margins: List[float] = []
        self._g3_margins: List[float] = []
        self._g4_margins: List[float] = []

        # Consecutive block tracking
        self._current_block_streak: int = 0
        self._max_block_streak: int = 0

        # Notable events
        self._notable_events: List[NotableEvent] = []
        self._first_block_recorded: bool = False

    def evaluate_cycle(self, input: WhatIfCycleInput) -> WhatIfCycleResult:
        """
        Evaluate a single cycle and emit hypothetical verdict.

        Args:
            input: Cycle input data

        Returns:
            WhatIfCycleResult with hypothetical verdict
        """
        # Evaluate G2: Invariant Gate
        g2_status, g2_reason = self._evaluate_g2(input)

        # Evaluate G3: Safe Region Gate
        g3_status, g3_reason = self._evaluate_g3(input)

        # Evaluate G4: Soft Gate (RSI)
        g4_status, g4_reason = self._evaluate_g4(input)

        # Determine verdict (G2 > G3 > G4 precedence)
        if g2_status == "FAIL":
            verdict = "BLOCK"
            blocking_gate = "G2_INVARIANT"
            blocking_reason = g2_reason
        elif g3_status == "FAIL":
            verdict = "BLOCK"
            blocking_gate = "G3_SAFE_REGION"
            blocking_reason = g3_reason
        elif g4_status == "FAIL":
            verdict = "BLOCK"
            blocking_gate = "G4_SOFT"
            blocking_reason = g4_reason
        else:
            verdict = "ALLOW"
            blocking_gate = None
            blocking_reason = None

        result = WhatIfCycleResult(
            cycle=input.cycle,
            timestamp=input.timestamp,
            verdict=verdict,
            blocking_gate=blocking_gate,
            blocking_reason=blocking_reason,
            g2_status=g2_status,
            g2_trigger=len(input.invariant_violations),
            g3_status=g3_status,
            g3_trigger=input.omega_exit_streak,
            g4_status=g4_status,
            g4_trigger=input.rho,
            g4_streak=input.rho_collapse_streak,
        )

        # Track results
        self._results.append(result)

        if verdict == "BLOCK":
            self._blocks.append(result)
            self._current_block_streak += 1
            self._max_block_streak = max(self._max_block_streak, self._current_block_streak)

            # Record first block as notable event
            if not self._first_block_recorded:
                self._notable_events.append(NotableEvent(
                    cycle=input.cycle,
                    event_type="FIRST_HYPOTHETICAL_BLOCK",
                    gate_id=blocking_gate,
                    description=blocking_reason,
                    hypothetical_verdict="BLOCK",
                ))
                self._first_block_recorded = True
        else:
            self._current_block_streak = 0

        # Update breach tracking
        self._update_breach_tracking(input, g2_status, g3_status, g4_status)

        # Update peak tracking
        self._update_peak_tracking(input)

        # Update margin tracking
        self._update_margin_tracking(input)

        return result

    def _evaluate_g2(self, input: WhatIfCycleInput) -> Tuple[str, Optional[str]]:
        """Evaluate G2: Invariant Gate."""
        violation_count = len(input.invariant_violations)

        if violation_count > self.config.invariant_tolerance:
            return "FAIL", f"Invariant violations: {input.invariant_violations}"
        return "PASS", None

    def _evaluate_g3(self, input: WhatIfCycleInput) -> Tuple[str, Optional[str]]:
        """Evaluate G3: Safe Region Gate."""
        if not input.in_omega and input.omega_exit_streak > self.config.omega_exit_threshold:
            return "FAIL", f"Outside safe region Ω for {input.omega_exit_streak} cycles (threshold: {self.config.omega_exit_threshold})"
        return "PASS", None

    def _evaluate_g4(self, input: WhatIfCycleInput) -> Tuple[str, Optional[str]]:
        """Evaluate G4: Soft Gate (RSI collapse)."""
        if input.rho < self.config.rho_min and input.rho_collapse_streak >= self.config.rho_streak_threshold:
            return "FAIL", f"RSI collapse (ρ={input.rho:.3f} < {self.config.rho_min} for {input.rho_collapse_streak} cycles)"
        return "PASS", None

    def _update_breach_tracking(
        self,
        input: WhatIfCycleInput,
        g2_status: str,
        g3_status: str,
        g4_status: str,
    ) -> None:
        """Track threshold breaches for analysis."""
        # G2 breach tracking
        if g2_status == "FAIL":
            if self._g2_breach_start is None:
                self._g2_breach_start = input.cycle
        else:
            if self._g2_breach_start is not None:
                duration = input.cycle - self._g2_breach_start
                self._g2_breaches.append(ThresholdBreach(
                    cycle=self._g2_breach_start,
                    trigger_value=len(input.invariant_violations),
                    duration=duration,
                ))
                self._g2_breach_start = None

        # G3 breach tracking
        if g3_status == "FAIL":
            if self._g3_breach_start is None:
                self._g3_breach_start = input.cycle
        else:
            if self._g3_breach_start is not None:
                duration = input.cycle - self._g3_breach_start
                self._g3_breaches.append(ThresholdBreach(
                    cycle=self._g3_breach_start,
                    trigger_value=input.omega_exit_streak,
                    duration=duration,
                ))
                self._g3_breach_start = None

        # G4 breach tracking
        if g4_status == "FAIL":
            if self._g4_breach_start is None:
                self._g4_breach_start = input.cycle
        else:
            if self._g4_breach_start is not None:
                duration = input.cycle - self._g4_breach_start
                self._g4_breaches.append(ThresholdBreach(
                    cycle=self._g4_breach_start,
                    trigger_value={"rho": input.rho},
                    duration=duration,
                ))
                self._g4_breach_start = None

    def _update_peak_tracking(self, input: WhatIfCycleInput) -> None:
        """Track peak trigger values."""
        # G2: Peak violation count
        violation_count = len(input.invariant_violations)
        if violation_count > self._g2_peak[0]:
            self._g2_peak = (violation_count, input.cycle)

        # G3: Peak omega exit streak
        if input.omega_exit_streak > self._g3_peak[0]:
            self._g3_peak = (input.omega_exit_streak, input.cycle)

        # G4: Peak RSI collapse (lower rho with higher streak is worse)
        if input.rho < self._g4_peak[0] or (
            input.rho == self._g4_peak[0] and input.rho_collapse_streak > self._g4_peak[1]
        ):
            self._g4_peak = (input.rho, input.rho_collapse_streak, input.cycle)

    def _update_margin_tracking(self, input: WhatIfCycleInput) -> None:
        """Track margins to threshold for mean computation."""
        # G2: Margin is violations below tolerance
        g2_margin = self.config.invariant_tolerance - len(input.invariant_violations)
        self._g2_margins.append(g2_margin)

        # G3: Margin is cycles below threshold
        g3_margin = self.config.omega_exit_threshold - input.omega_exit_streak
        self._g3_margins.append(g3_margin)

        # G4: Margin is rho above minimum
        g4_margin = input.rho - self.config.rho_min
        self._g4_margins.append(g4_margin)

    def build_report(self) -> WhatIfReport:
        """
        Build complete What-If analysis report.

        Returns:
            WhatIfReport matching schema specification
        """
        total_cycles = len(self._results)
        hypothetical_blocks = len(self._blocks)
        hypothetical_allows = total_cycles - hypothetical_blocks

        # Close any open breaches
        self._close_open_breaches()

        # Build gate distribution
        gate_distribution: Dict[str, int] = {}
        for block in self._blocks:
            gate = block.blocking_gate
            gate_distribution[gate] = gate_distribution.get(gate, 0) + 1

        # Build gate analyses
        g2_analysis = self._build_g2_analysis(total_cycles)
        g3_analysis = self._build_g3_analysis(total_cycles)
        g4_analysis = self._build_g4_analysis(total_cycles)

        # Generate calibration recommendations
        recommendations = self._generate_recommendations()

        # Build auditor notes
        auditor_notes = self._generate_auditor_notes(
            total_cycles, hypothetical_blocks, gate_distribution
        )

        return WhatIfReport(
            schema_version="1.0.0",
            run_id=self.run_id,
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            mode="HYPOTHETICAL",
            total_cycles=total_cycles,
            hypothetical_allows=hypothetical_allows,
            hypothetical_blocks=hypothetical_blocks,
            hypothetical_block_rate=hypothetical_blocks / total_cycles if total_cycles > 0 else 0.0,
            blocking_gate_distribution=gate_distribution,
            max_consecutive_blocks=self._max_block_streak,
            first_hypothetical_block_cycle=self._blocks[0].cycle if self._blocks else None,
            g2_analysis=g2_analysis,
            g3_analysis=g3_analysis,
            g4_analysis=g4_analysis,
            notable_events=self._notable_events,
            calibration_recommendations=recommendations,
            auditor_notes=auditor_notes,
        )

    def _close_open_breaches(self) -> None:
        """Close any breaches still open at end of run."""
        if self._results:
            last_cycle = self._results[-1].cycle
            last_input = WhatIfCycleInput(cycle=last_cycle, timestamp="")

            if self._g2_breach_start is not None:
                duration = last_cycle - self._g2_breach_start + 1
                self._g2_breaches.append(ThresholdBreach(
                    cycle=self._g2_breach_start,
                    trigger_value=self._g2_peak[0],
                    duration=duration,
                ))

            if self._g3_breach_start is not None:
                duration = last_cycle - self._g3_breach_start + 1
                self._g3_breaches.append(ThresholdBreach(
                    cycle=self._g3_breach_start,
                    trigger_value=self._g3_peak[0],
                    duration=duration,
                ))

            if self._g4_breach_start is not None:
                duration = last_cycle - self._g4_breach_start + 1
                self._g4_breaches.append(ThresholdBreach(
                    cycle=self._g4_breach_start,
                    trigger_value={"rho": self._g4_peak[0]},
                    duration=duration,
                ))

    def _build_g2_analysis(self, total_cycles: int) -> GateWhatIfAnalysis:
        """Build G2 gate analysis."""
        fail_count = sum(1 for r in self._results if r.g2_status == "FAIL")
        block_count = sum(1 for r in self._blocks if r.blocking_gate == "G2_INVARIANT")

        return GateWhatIfAnalysis(
            gate_id="G2_INVARIANT",
            hypothetical_fail_count=fail_count,
            hypothetical_block_count=block_count,
            fail_rate=fail_count / total_cycles if total_cycles > 0 else 0.0,
            peak_trigger_value=self._g2_peak[0] if self._g2_peak[0] > 0 else None,
            peak_cycle=self._g2_peak[1] if self._g2_peak[0] > 0 else None,
            mean_margin_to_threshold=sum(self._g2_margins) / len(self._g2_margins) if self._g2_margins else 0.0,
            threshold_breaches=self._g2_breaches,
        )

    def _build_g3_analysis(self, total_cycles: int) -> GateWhatIfAnalysis:
        """Build G3 gate analysis."""
        fail_count = sum(1 for r in self._results if r.g3_status == "FAIL")
        block_count = sum(1 for r in self._blocks if r.blocking_gate == "G3_SAFE_REGION")

        return GateWhatIfAnalysis(
            gate_id="G3_SAFE_REGION",
            hypothetical_fail_count=fail_count,
            hypothetical_block_count=block_count,
            fail_rate=fail_count / total_cycles if total_cycles > 0 else 0.0,
            peak_trigger_value=self._g3_peak[0] if self._g3_peak[0] > 0 else None,
            peak_cycle=self._g3_peak[1] if self._g3_peak[0] > 0 else None,
            mean_margin_to_threshold=sum(self._g3_margins) / len(self._g3_margins) if self._g3_margins else 0.0,
            threshold_breaches=self._g3_breaches,
        )

    def _build_g4_analysis(self, total_cycles: int) -> GateWhatIfAnalysis:
        """Build G4 gate analysis."""
        fail_count = sum(1 for r in self._results if r.g4_status == "FAIL")
        block_count = sum(1 for r in self._blocks if r.blocking_gate == "G4_SOFT")

        peak_value = None
        if self._g4_peak[1] > 0:  # Has meaningful streak
            peak_value = {"rho": self._g4_peak[0], "streak": self._g4_peak[1]}

        return GateWhatIfAnalysis(
            gate_id="G4_SOFT",
            hypothetical_fail_count=fail_count,
            hypothetical_block_count=block_count,
            fail_rate=fail_count / total_cycles if total_cycles > 0 else 0.0,
            peak_trigger_value=peak_value,
            peak_cycle=self._g4_peak[2] if self._g4_peak[1] > 0 else None,
            mean_margin_to_threshold=sum(self._g4_margins) / len(self._g4_margins) if self._g4_margins else 0.0,
            threshold_breaches=self._g4_breaches,
        )

    def _generate_recommendations(self) -> List[CalibrationRecommendation]:
        """Generate calibration recommendations based on analysis."""
        recommendations = []

        # G4 RSI recommendation if many transient failures
        g4_blocks = sum(1 for r in self._blocks if r.blocking_gate == "G4_SOFT")
        if g4_blocks > 0 and self._g4_breaches:
            avg_breach_duration = sum(b.duration for b in self._g4_breaches) / len(self._g4_breaches)
            if avg_breach_duration < 20:  # Short breaches suggest transient issues
                recommendations.append(CalibrationRecommendation(
                    gate_id="G4_SOFT",
                    parameter="rho_streak_threshold",
                    current_value=self.config.rho_streak_threshold,
                    suggested_value=self.config.rho_streak_threshold + 5,
                    rationale=f"Average breach duration ({avg_breach_duration:.1f} cycles) suggests transient RSI dips; increase threshold to allow recovery",
                    confidence="MEDIUM",
                ))

        # G3 recommendation if omega exits are brief
        g3_blocks = sum(1 for r in self._blocks if r.blocking_gate == "G3_SAFE_REGION")
        if g3_blocks > 0 and self._g3_breaches:
            avg_breach_duration = sum(b.duration for b in self._g3_breaches) / len(self._g3_breaches)
            if avg_breach_duration < 30:
                recommendations.append(CalibrationRecommendation(
                    gate_id="G3_SAFE_REGION",
                    parameter="omega_exit_threshold",
                    current_value=self.config.omega_exit_threshold,
                    suggested_value=self.config.omega_exit_threshold + 20,
                    rationale=f"Average Omega exit duration ({avg_breach_duration:.1f} cycles) suggests transient excursions; consider increasing threshold",
                    confidence="LOW",
                ))

        return recommendations

    def _generate_auditor_notes(
        self,
        total_cycles: int,
        hypothetical_blocks: int,
        gate_distribution: Dict[str, int],
    ) -> str:
        """Generate auditor notes summary."""
        if total_cycles == 0:
            return "No cycles evaluated."

        block_rate = hypothetical_blocks / total_cycles * 100

        parts = [
            f"HYPOTHETICAL ANALYSIS ONLY.",
            f"Run exhibited {block_rate:.1f}% hypothetical block rate ({hypothetical_blocks}/{total_cycles} cycles).",
        ]

        if gate_distribution:
            most_common = max(gate_distribution.items(), key=lambda x: x[1])
            parts.append(f"{most_common[0]} was the most frequent blocking condition ({most_common[1]} blocks).")

        parts.append("No enforcement action taken—SHADOW MODE.")

        return " ".join(parts)

    def get_results(self) -> List[WhatIfCycleResult]:
        """Get all cycle results."""
        return self._results

    def get_blocks(self) -> List[WhatIfCycleResult]:
        """Get only BLOCK results."""
        return self._blocks

    def reset(self) -> None:
        """Reset engine state."""
        self._results.clear()
        self._blocks.clear()
        self._g2_breach_start = None
        self._g3_breach_start = None
        self._g4_breach_start = None
        self._g2_breaches.clear()
        self._g3_breaches.clear()
        self._g4_breaches.clear()
        self._g2_peak = (0, 0)
        self._g3_peak = (0, 0)
        self._g4_peak = (1.0, 0, 0)
        self._g2_margins.clear()
        self._g3_margins.clear()
        self._g4_margins.clear()
        self._current_block_streak = 0
        self._max_block_streak = 0
        self._notable_events.clear()
        self._first_block_recorded = False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def build_what_if_report(
    telemetry: List[Dict[str, Any]],
    config: Optional[WhatIfConfig] = None,
    run_id: Optional[str] = None,
) -> WhatIfReport:
    """
    Build What-If report from telemetry data.

    Args:
        telemetry: List of cycle telemetry dictionaries
        config: Optional configuration
        run_id: Optional run identifier

    Returns:
        WhatIfReport
    """
    engine = WhatIfEngine(config=config, run_id=run_id)

    for i, data in enumerate(telemetry):
        cycle = data.get("cycle", i + 1)
        input = WhatIfCycleInput.from_telemetry(data, cycle)
        engine.evaluate_cycle(input)

    return engine.build_report()


def export_what_if_report(report: WhatIfReport, path: str) -> None:
    """Export What-If report to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
