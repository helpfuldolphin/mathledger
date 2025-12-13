"""
Phase X P5: TDA Divergence Pattern Classifier

This module implements the TDAPatternClassifier for classifying P5 divergence
patterns into the 6-pattern taxonomy defined in Real_Telemetry_Topology_Spec.md.

See:
- docs/system_law/Real_Telemetry_Topology_Spec.md Section 3
- docs/system_law/Global_Governance_Fusion_PhaseX.md Section 11
- docs/system_law/GGFL_P5_Pattern_Test_Plan.md

SHADOW MODE CONTRACT:
1. Classification is for LOGGING and ANALYSIS only
2. Pattern detection does NOT trigger any governance enforcement
3. All outputs have mode="SHADOW"
4. No control flow depends on classification results

Status: SPEC→CODE (Phase X SHADOW MODE)
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "DivergencePattern",
    "PatternClassification",
    "TDAPatternClassifier",
    "P5TelemetryExtension",
    "P5TopologyExtension",
    "P5ReplayExtension",
    "attach_tda_patterns_to_evidence",
    "P5_PATTERN_SCHEMA_VERSION",
]

# Schema version for P5 pattern classification output
P5_PATTERN_SCHEMA_VERSION = "1.0.0"


class DivergencePattern(str, Enum):
    """
    P5 Divergence Pattern Taxonomy.

    See Real_Telemetry_Topology_Spec.md Section 3.1
    """
    DRIFT = "DRIFT"
    NOISE_AMPLIFICATION = "NOISE_AMPLIFICATION"
    PHASE_LAG = "PHASE_LAG"
    ATTRACTOR_MISS = "ATTRACTOR_MISS"
    TRANSIENT_MISS = "TRANSIENT_MISS"
    STRUCTURAL_BREAK = "STRUCTURAL_BREAK"
    UNCLASSIFIED = "UNCLASSIFIED"
    NOMINAL = "NOMINAL"


@dataclass
class PatternThresholds:
    """
    Thresholds for P5 divergence pattern classification.

    SHADOW MODE: These thresholds are for classification only.
    They do NOT trigger enforcement.
    """
    # DRIFT thresholds
    drift_mean_threshold: float = 0.05
    drift_std_threshold: float = 0.02

    # NOISE_AMPLIFICATION thresholds
    noise_std_multiplier: float = 2.0

    # PHASE_LAG thresholds (not used directly in simplified classifier)
    phase_lag_threshold: int = 0

    # ATTRACTOR_MISS thresholds
    attractor_miss_rate_threshold: float = 0.10

    # TRANSIENT_MISS thresholds
    transient_miss_ratio: float = 2.0

    # STRUCTURAL_BREAK thresholds
    structural_break_delta_threshold: float = 0.10
    structural_break_recent_threshold: float = 0.05


@dataclass
class PatternClassification:
    """
    Result of pattern classification.

    SHADOW MODE: Classification is observational only.
    """
    pattern: DivergencePattern
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern": self.pattern.value,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
            "timestamp": self.timestamp,
        }


@dataclass
class P5TelemetryExtension:
    """
    P5 extension fields for SIG-TEL.

    See Global_Governance_Fusion_PhaseX.md Section 11.2.1
    """
    telemetry_validation_status: str = "VALIDATION_PENDING"
    validation_confidence: float = 0.0
    divergence_pattern: str = "NOMINAL"
    divergence_pattern_streak: int = 0
    recalibration_triggered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "telemetry_validation_status": self.telemetry_validation_status,
            "validation_confidence": round(self.validation_confidence, 4),
            "divergence_pattern": self.divergence_pattern,
            "divergence_pattern_streak": self.divergence_pattern_streak,
            "recalibration_triggered": self.recalibration_triggered,
        }


@dataclass
class P5TopologyExtension:
    """
    P5 extension fields for SIG-TOP.

    See Global_Governance_Fusion_PhaseX.md Section 11.2.2
    """
    attractor_miss_rate: float = 0.0
    twin_omega_alignment: bool = True
    transient_tracking_quality: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attractor_miss_rate": round(self.attractor_miss_rate, 4),
            "twin_omega_alignment": self.twin_omega_alignment,
            "transient_tracking_quality": round(self.transient_tracking_quality, 4),
        }


@dataclass
class P5ReplayExtension:
    """
    P5 extension fields for SIG-RPL.

    See Global_Governance_Fusion_PhaseX.md Section 11.2.3
    """
    twin_prediction_divergence: float = 0.0
    divergence_bias: float = 0.0
    divergence_variance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "twin_prediction_divergence": round(self.twin_prediction_divergence, 4),
            "divergence_bias": round(self.divergence_bias, 4),
            "divergence_variance": round(self.divergence_variance, 6),
        }


class TDAPatternClassifier:
    """
    Classifier for P5 divergence patterns.

    SHADOW MODE CONTRACT:
    - Classification is for LOGGING and ANALYSIS only
    - Pattern detection does NOT trigger any governance enforcement
    - All methods are observation-only
    - streak tracking is for analytics, not control

    Pattern Definitions (from Real_Telemetry_Topology_Spec.md Section 3.1):

    | Pattern | Signature |
    |---------|-----------|
    | DRIFT | mean(Δp) > 0.05, std(Δp) < 0.02 |
    | NOISE_AMPLIFICATION | std(Δp) > 2 × std(p_real) |
    | PHASE_LAG | argmax(xcorr(p_twin, p_real)) ≠ 0 |
    | ATTRACTOR_MISS | ω_twin ≠ ω_real frequently |
    | TRANSIENT_MISS | High Δp during excursions only |
    | STRUCTURAL_BREAK | Δp suddenly increases, stays high |
    """

    def __init__(
        self,
        thresholds: Optional[PatternThresholds] = None,
        window_size: int = 50,
    ) -> None:
        """
        Initialize the pattern classifier.

        Args:
            thresholds: Pattern classification thresholds
            window_size: Rolling window size for pattern detection
        """
        self._thresholds = thresholds or PatternThresholds()
        self._window_size = window_size

        # Rolling history for pattern detection
        self._delta_history: List[float] = []
        self._p_real_history: List[float] = []
        self._omega_real_history: List[bool] = []
        self._omega_twin_history: List[bool] = []

        # Streak tracking
        self._current_pattern: DivergencePattern = DivergencePattern.NOMINAL
        self._pattern_streak: int = 0
        self._last_classification: Optional[PatternClassification] = None

    def classify(
        self,
        delta_p: float,
        p_real: float,
        p_twin: float,
        omega_real: bool,
        omega_twin: bool,
        is_excursion: bool = False,
    ) -> PatternClassification:
        """
        Classify the current divergence pattern.

        SHADOW MODE: Classification is for logging only.

        Args:
            delta_p: Divergence value (p_twin - p_real)
            p_real: Real telemetry value
            p_twin: Twin prediction value
            omega_real: Real safe region membership
            omega_twin: Twin predicted safe region
            is_excursion: Whether this is a transient excursion

        Returns:
            PatternClassification with pattern and confidence
        """
        # Update rolling histories
        self._delta_history.append(delta_p)
        self._p_real_history.append(p_real)
        self._omega_real_history.append(omega_real)
        self._omega_twin_history.append(omega_twin)

        # Trim to window size
        if len(self._delta_history) > self._window_size:
            self._delta_history = self._delta_history[-self._window_size:]
            self._p_real_history = self._p_real_history[-self._window_size:]
            self._omega_real_history = self._omega_real_history[-self._window_size:]
            self._omega_twin_history = self._omega_twin_history[-self._window_size:]

        # Need minimum samples for classification
        if len(self._delta_history) < 10:
            classification = PatternClassification(
                pattern=DivergencePattern.NOMINAL,
                confidence=0.5,
                evidence={"reason": "insufficient_samples"},
            )
            self._update_streak(classification.pattern)
            self._last_classification = classification
            return classification

        # Compute statistics
        mean_delta = statistics.mean(self._delta_history)
        std_delta = statistics.stdev(self._delta_history) if len(self._delta_history) > 1 else 0.0
        std_real = statistics.stdev(self._p_real_history) if len(self._p_real_history) > 1 else 0.001

        # Omega miss rate
        omega_misses = sum(
            1 for r, t in zip(self._omega_real_history, self._omega_twin_history)
            if r != t
        )
        omega_miss_rate = omega_misses / len(self._omega_real_history)

        # Check patterns in priority order
        classification = self._classify_pattern(
            mean_delta=mean_delta,
            std_delta=std_delta,
            std_real=std_real,
            omega_miss_rate=omega_miss_rate,
            is_excursion=is_excursion,
            delta_p=delta_p,
        )

        self._update_streak(classification.pattern)
        self._last_classification = classification
        return classification

    def _classify_pattern(
        self,
        mean_delta: float,
        std_delta: float,
        std_real: float,
        omega_miss_rate: float,
        is_excursion: bool,
        delta_p: float,
    ) -> PatternClassification:
        """
        Apply pattern classification rules.

        Returns:
            PatternClassification with dominant pattern
        """
        th = self._thresholds
        evidence: Dict[str, Any] = {
            "mean_delta": round(mean_delta, 4),
            "std_delta": round(std_delta, 4),
            "std_real": round(std_real, 4),
            "omega_miss_rate": round(omega_miss_rate, 4),
            "is_excursion": is_excursion,
        }

        # 1. STRUCTURAL_BREAK: Sudden increase, stays high
        if self._detect_structural_break(delta_p, mean_delta):
            return PatternClassification(
                pattern=DivergencePattern.STRUCTURAL_BREAK,
                confidence=0.9,
                evidence={**evidence, "trigger": "sudden_increase_sustained"},
            )

        # 2. ATTRACTOR_MISS: Frequent omega mismatch
        if omega_miss_rate > th.attractor_miss_rate_threshold:
            return PatternClassification(
                pattern=DivergencePattern.ATTRACTOR_MISS,
                confidence=min(0.95, 0.5 + omega_miss_rate),
                evidence={**evidence, "trigger": "omega_mismatch"},
            )

        # 3. TRANSIENT_MISS: High delta during excursions only
        if is_excursion and abs(delta_p) > th.drift_mean_threshold * 2:
            # Check if non-excursion deltas are low
            if std_delta < th.drift_std_threshold * 2:
                return PatternClassification(
                    pattern=DivergencePattern.TRANSIENT_MISS,
                    confidence=0.8,
                    evidence={**evidence, "trigger": "excursion_only_divergence"},
                )

        # 4. DRIFT: Systematic bias
        if abs(mean_delta) > th.drift_mean_threshold and std_delta < th.drift_std_threshold:
            return PatternClassification(
                pattern=DivergencePattern.DRIFT,
                confidence=0.85,
                evidence={**evidence, "trigger": "systematic_bias"},
            )

        # 5. NOISE_AMPLIFICATION: Twin over-sensitive
        if std_delta > th.noise_std_multiplier * std_real and std_real > 0.001:
            return PatternClassification(
                pattern=DivergencePattern.NOISE_AMPLIFICATION,
                confidence=0.75,
                evidence={
                    **evidence,
                    "trigger": "twin_oversensitive",
                    "std_ratio": round(std_delta / std_real, 2),
                },
            )

        # 6. PHASE_LAG: Simplified check - twin systematically behind
        # (Full xcorr not implemented; use simplified heuristic)
        if self._detect_phase_lag():
            return PatternClassification(
                pattern=DivergencePattern.PHASE_LAG,
                confidence=0.7,
                evidence={**evidence, "trigger": "temporal_misalignment"},
            )

        # No pattern detected
        return PatternClassification(
            pattern=DivergencePattern.NOMINAL,
            confidence=0.8,
            evidence={**evidence, "trigger": "none"},
        )

    def _detect_structural_break(
        self,
        current_delta: float,
        mean_delta: float,
    ) -> bool:
        """
        Detect structural break pattern.

        Signature: Δp suddenly increases and stays high.
        """
        if len(self._delta_history) < 20:
            return False

        th = self._thresholds

        # Check recent vs historical
        recent = self._delta_history[-10:]
        historical = self._delta_history[-20:-10]

        recent_mean = abs(statistics.mean(recent))
        historical_mean = abs(statistics.mean(historical))

        # Sudden increase
        if recent_mean > historical_mean + th.structural_break_delta_threshold:
            # Check it stays high
            if all(abs(d) > th.structural_break_recent_threshold for d in recent[-5:]):
                return True

        return False

    def _detect_phase_lag(self) -> bool:
        """
        Simplified phase lag detection.

        Full cross-correlation not implemented; use sign-change heuristic.
        """
        if len(self._delta_history) < 20:
            return False

        # Check for systematic sign pattern in deltas
        # (Phase lag often shows alternating positive/negative)
        sign_changes = sum(
            1 for i in range(1, len(self._delta_history))
            if (self._delta_history[i] > 0) != (self._delta_history[i-1] > 0)
        )

        # High sign change rate suggests temporal misalignment
        sign_change_rate = sign_changes / len(self._delta_history)
        return sign_change_rate > 0.4

    def _update_streak(self, pattern: DivergencePattern) -> None:
        """Update pattern streak tracking."""
        if pattern == self._current_pattern:
            self._pattern_streak += 1
        else:
            self._current_pattern = pattern
            self._pattern_streak = 1

    def get_current_pattern(self) -> DivergencePattern:
        """Get the current dominant pattern."""
        return self._current_pattern

    def get_pattern_streak(self) -> int:
        """Get the current pattern streak length."""
        return self._pattern_streak

    def get_p5_telemetry_extension(
        self,
        validation_status: str = "VALIDATED_REAL",
        validation_confidence: float = 0.9,
    ) -> P5TelemetryExtension:
        """
        Build P5 telemetry extension for SIG-TEL.

        Args:
            validation_status: Telemetry validation status
            validation_confidence: Validation confidence

        Returns:
            P5TelemetryExtension for GGFL integration
        """
        return P5TelemetryExtension(
            telemetry_validation_status=validation_status,
            validation_confidence=validation_confidence,
            divergence_pattern=self._current_pattern.value,
            divergence_pattern_streak=self._pattern_streak,
            recalibration_triggered=self._should_trigger_recalibration(),
        )

    def get_p5_topology_extension(self) -> P5TopologyExtension:
        """
        Build P5 topology extension for SIG-TOP.

        Returns:
            P5TopologyExtension for GGFL integration
        """
        # Compute attractor miss rate
        if len(self._omega_real_history) > 0:
            misses = sum(
                1 for r, t in zip(self._omega_real_history, self._omega_twin_history)
                if r != t
            )
            miss_rate = misses / len(self._omega_real_history)
        else:
            miss_rate = 0.0

        # Twin omega alignment
        alignment = miss_rate < 0.1

        # Transient tracking quality (inverse of transient divergence)
        quality = max(0.0, 1.0 - miss_rate * 2)

        return P5TopologyExtension(
            attractor_miss_rate=miss_rate,
            twin_omega_alignment=alignment,
            transient_tracking_quality=quality,
        )

    def get_p5_replay_extension(self) -> P5ReplayExtension:
        """
        Build P5 replay extension for SIG-RPL.

        Returns:
            P5ReplayExtension for GGFL integration
        """
        if len(self._delta_history) == 0:
            return P5ReplayExtension()

        mean_delta = statistics.mean(self._delta_history)
        variance = statistics.variance(self._delta_history) if len(self._delta_history) > 1 else 0.0

        return P5ReplayExtension(
            twin_prediction_divergence=abs(mean_delta),
            divergence_bias=mean_delta,
            divergence_variance=variance,
        )

    def _should_trigger_recalibration(self) -> bool:
        """
        Check if recalibration would be triggered.

        SHADOW MODE: This is observational only.

        Triggers:
        - Rolling mean(|Δp|) > 0.10 for 3+ windows
        - Pattern = STRUCTURAL_BREAK
        """
        if self._current_pattern == DivergencePattern.STRUCTURAL_BREAK:
            return True

        if len(self._delta_history) >= 30:
            recent_mean = statistics.mean(abs(d) for d in self._delta_history[-30:])
            if recent_mean > 0.10:
                return True

        return False

    def reset(self) -> None:
        """Reset classifier state."""
        self._delta_history.clear()
        self._p_real_history.clear()
        self._omega_real_history.clear()
        self._omega_twin_history.clear()
        self._current_pattern = DivergencePattern.NOMINAL
        self._pattern_streak = 0
        self._last_classification = None


def attach_tda_patterns_to_evidence(
    evidence: Dict[str, Any],
    classifier: TDAPatternClassifier,
    validation_status: str = "VALIDATED_REAL",
    validation_confidence: float = 0.9,
) -> Dict[str, Any]:
    """
    Attach P5 TDA pattern classification to evidence pack.

    SHADOW MODE CONTRACT:
    - Attached data is for ANALYSIS only
    - Does NOT modify governance decisions
    - All outputs include mode="SHADOW"

    Args:
        evidence: Evidence pack dict to augment
        classifier: TDAPatternClassifier with accumulated state
        validation_status: Telemetry validation status
        validation_confidence: Validation confidence score

    Returns:
        Evidence dict with p5_pattern_classification attached
    """
    # Build classification snapshot
    p5_classification = {
        "schema_version": P5_PATTERN_SCHEMA_VERSION,
        "mode": "SHADOW",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "classification": {
            "current_pattern": classifier.get_current_pattern().value,
            "pattern_streak": classifier.get_pattern_streak(),
            "last_classification": (
                classifier._last_classification.to_dict()
                if classifier._last_classification
                else None
            ),
        },
        "signal_extensions": {
            "p5_telemetry": classifier.get_p5_telemetry_extension(
                validation_status=validation_status,
                validation_confidence=validation_confidence,
            ).to_dict(),
            "p5_topology": classifier.get_p5_topology_extension().to_dict(),
            "p5_replay": classifier.get_p5_replay_extension().to_dict(),
        },
        "recalibration_triggered": classifier._should_trigger_recalibration(),
        "shadow_mode_invariants": {
            "no_enforcement": True,
            "logged_only": True,
            "observation_only": True,
        },
    }

    # Attach to evidence under governance.p5_pattern_classification
    if "governance" not in evidence:
        evidence["governance"] = {}
    evidence["governance"]["p5_pattern_classification"] = p5_classification

    return evidence


# =============================================================================
# Direct pattern classification from signal dict (for GGFL integration)
# =============================================================================

def classify_from_signals(
    p5_telemetry: Optional[Dict[str, Any]] = None,
    p5_topology: Optional[Dict[str, Any]] = None,
    p5_replay: Optional[Dict[str, Any]] = None,
) -> PatternClassification:
    """
    Classify pattern directly from P5 signal extension dicts.

    This is used when signals are already assembled (e.g., from test fixtures).

    SHADOW MODE: Classification is for logging only.

    Args:
        p5_telemetry: P5 telemetry extension dict
        p5_topology: P5 topology extension dict
        p5_replay: P5 replay extension dict

    Returns:
        PatternClassification based on signal values
    """
    if p5_telemetry is None:
        return PatternClassification(
            pattern=DivergencePattern.NOMINAL,
            confidence=0.5,
            evidence={"reason": "no_p5_telemetry"},
        )

    # Extract pattern from telemetry (already classified)
    pattern_str = p5_telemetry.get("divergence_pattern", "NOMINAL")
    streak = p5_telemetry.get("divergence_pattern_streak", 0)

    try:
        pattern = DivergencePattern(pattern_str)
    except ValueError:
        pattern = DivergencePattern.UNCLASSIFIED

    # Build evidence from available extensions
    evidence: Dict[str, Any] = {
        "source": "signal_extensions",
        "streak": streak,
    }

    if p5_topology:
        evidence["attractor_miss_rate"] = p5_topology.get("attractor_miss_rate", 0.0)
        evidence["twin_omega_alignment"] = p5_topology.get("twin_omega_alignment", True)

    if p5_replay:
        evidence["twin_prediction_divergence"] = p5_replay.get("twin_prediction_divergence", 0.0)
        evidence["divergence_bias"] = p5_replay.get("divergence_bias", 0.0)

    # Confidence based on pattern and streak
    confidence = 0.8
    if streak >= 5:
        confidence = 0.9
    if pattern == DivergencePattern.STRUCTURAL_BREAK and streak >= 2:
        confidence = 0.95

    return PatternClassification(
        pattern=pattern,
        confidence=confidence,
        evidence=evidence,
    )
