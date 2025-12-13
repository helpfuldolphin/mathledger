"""
TDA Pattern Classifier

Classifies RTTS divergence patterns from TDA metrics with deterministic
priority ordering and confidence scoring.

See: docs/system_law/TDA_PhaseX_Binding.md Section 10

SHADOW MODE CONTRACT:
- All classifications are observational only
- No governance modification based on pattern detection
- Classifications are logged for analysis, never enforced
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from backend.tda.monitor import TDASummary
from backend.tda.metrics import TDAWindowMetrics

__all__ = [
    "RTTSPattern",
    "PatternClassification",
    "TDAPatternClassifier",
    "attach_tda_patterns_to_evidence",
]


class RTTSPattern(Enum):
    """RTTS divergence patterns classifiable from TDA metrics."""
    NONE = "NONE"
    DRIFT = "DRIFT"
    PHASE_LAG = "PHASE_LAG"
    STRUCTURAL_BREAK = "STRUCTURAL_BREAK"
    NOISE_AMPLIFICATION = "NOISE_AMPLIFICATION"
    ATTRACTOR_MISS = "ATTRACTOR_MISS"
    TRANSIENT_MISS = "TRANSIENT_MISS"


# Priority ordering: lower index = higher priority
PATTERN_PRIORITY = [
    RTTSPattern.STRUCTURAL_BREAK,
    RTTSPattern.ATTRACTOR_MISS,
    RTTSPattern.NOISE_AMPLIFICATION,
    RTTSPattern.PHASE_LAG,
    RTTSPattern.DRIFT,
    RTTSPattern.TRANSIENT_MISS,
    RTTSPattern.NONE,
]


@dataclass
class PatternClassification:
    """
    Result of TDA pattern classification.

    SHADOW MODE: Classification is observational only.
    """
    pattern: RTTSPattern
    confidence: float  # [0.0, 1.0]
    primary_triggers: List[str] = field(default_factory=list)
    secondary_triggers: List[str] = field(default_factory=list)
    metrics_snapshot: Dict[str, float] = field(default_factory=dict)
    window_id: Optional[str] = None
    cycle_range: Optional[Tuple[int, int]] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    mode: str = "SHADOW"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "pattern": self.pattern.value,
            "confidence": round(self.confidence, 4),
            "primary_triggers": self.primary_triggers,
            "secondary_triggers": self.secondary_triggers,
            "metrics_snapshot": {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in self.metrics_snapshot.items()
            },
            "window_id": self.window_id,
            "cycle_range": list(self.cycle_range) if self.cycle_range else None,
            "timestamp": self.timestamp,
            "mode": self.mode,
        }


class TDAPatternClassifier:
    """
    Classifies RTTS divergence patterns from TDA metrics.

    SHADOW MODE: Observational only. Classifications logged but not enforced.

    Usage:
        classifier = TDAPatternClassifier()
        result = classifier.classify(p3_tda=p3_summary, p4_tda=p4_summary)
        # result.pattern -> RTTSPattern.DRIFT
        # result.confidence -> 0.85
    """

    # ==========================================================================
    # Threshold Constants (from Section 10.1)
    # ==========================================================================

    # DRIFT thresholds
    DRIFT_DRS_THRESHOLD = 0.05
    DRIFT_DRS_SLOPE_THRESHOLD = 0.01
    DRIFT_SNS_CEILING = 0.4
    DRIFT_HSS_FLOOR = 0.6
    DRIFT_PCS_FLOOR = 0.5
    DRIFT_SUSTAINED_CYCLES = 3

    # PHASE_LAG thresholds
    PHASE_LAG_PCS_THRESHOLD = 0.5
    PHASE_LAG_PCS_DELTA_THRESHOLD = -0.15
    PHASE_LAG_DRS_MIN = 0.03
    PHASE_LAG_DRS_MAX = 0.10
    PHASE_LAG_HSS_FLOOR = 0.5
    PHASE_LAG_SNS_CEILING = 0.5

    # STRUCTURAL_BREAK thresholds
    STRUCTURAL_BREAK_HSS_THRESHOLD = 0.5
    STRUCTURAL_BREAK_HSS_DELTA_THRESHOLD = -0.25
    STRUCTURAL_BREAK_SNS_THRESHOLD = 0.5
    STRUCTURAL_BREAK_DRS_THRESHOLD = 0.08

    # NOISE_AMPLIFICATION thresholds
    NOISE_AMP_SNS_VARIANCE_THRESHOLD = 0.04
    NOISE_AMP_SNS_MEAN_THRESHOLD = 0.35
    NOISE_AMP_SNS_MAX_THRESHOLD = 0.6
    NOISE_AMP_PCS_CEILING = 0.6
    NOISE_AMP_HSS_FLOOR = 0.4
    NOISE_AMP_ENVELOPE_EXITS_THRESHOLD = 2

    # ATTRACTOR_MISS thresholds
    ATTRACTOR_MISS_HSS_THRESHOLD = 0.6
    ATTRACTOR_MISS_DRS_THRESHOLD = 0.10
    ATTRACTOR_MISS_SNS_MIN = 0.3
    ATTRACTOR_MISS_SNS_MAX = 0.6
    ATTRACTOR_MISS_PCS_CEILING = 0.7

    # TRANSIENT_MISS thresholds
    TRANSIENT_MISS_SNS_PEAK_THRESHOLD = 0.5
    TRANSIENT_MISS_SNS_RECOVERY_THRESHOLD = 0.3
    TRANSIENT_MISS_HSS_FLOOR = 0.6
    TRANSIENT_MISS_DRS_CEILING = 0.08
    TRANSIENT_MISS_RECOVERY_CYCLES = 3

    # Confidence threshold for pattern acceptance
    CONFIDENCE_THRESHOLD = 0.5

    def __init__(self) -> None:
        """Initialize TDA pattern classifier."""
        self._classification_count = 0

    def classify(
        self,
        p3_tda: Optional[TDASummary] = None,
        p4_tda: Optional[TDASummary] = None,
        window_history: Optional[List[TDAWindowMetrics]] = None,
        window_id: Optional[str] = None,
        cycle_range: Optional[Tuple[int, int]] = None,
    ) -> PatternClassification:
        """
        Classify RTTS pattern from TDA metrics.

        Args:
            p3_tda: P3 (First-Light synthetic) TDA summary
            p4_tda: P4 (shadow coupling) TDA summary
            window_history: Recent window metrics for trend analysis
            window_id: Optional window identifier
            cycle_range: Optional (start_cycle, end_cycle) tuple

        Returns:
            PatternClassification with pattern, confidence, and triggers

        Priority order (highest to lowest):
            1. STRUCTURAL_BREAK (catastrophic topology change)
            2. ATTRACTOR_MISS (convergence failure)
            3. NOISE_AMPLIFICATION (instability regime)
            4. PHASE_LAG (timing offset)
            5. DRIFT (gradual deviation)
            6. TRANSIENT_MISS (temporary excursion)
            7. NONE (stable state)
        """
        self._classification_count += 1

        # Step 1: Merge metrics from P3 and P4 sources
        metrics = self._merge_metrics(p3_tda, p4_tda)

        # Step 2: Compute deltas if history available
        if window_history and len(window_history) >= 2:
            metrics.update(self._compute_deltas(window_history))

        # Step 3: Check patterns in priority order
        candidates: List[Tuple[RTTSPattern, float, Dict[str, List[str]]]] = []

        # Priority 1: STRUCTURAL_BREAK
        match, conf, triggers = self._check_structural_break(metrics, window_history)
        if match:
            candidates.append((RTTSPattern.STRUCTURAL_BREAK, conf, triggers))

        # Priority 2: ATTRACTOR_MISS
        match, conf, triggers = self._check_attractor_miss(metrics, window_history)
        if match:
            candidates.append((RTTSPattern.ATTRACTOR_MISS, conf, triggers))

        # Priority 3: NOISE_AMPLIFICATION
        match, conf, triggers = self._check_noise_amplification(metrics, window_history)
        if match:
            candidates.append((RTTSPattern.NOISE_AMPLIFICATION, conf, triggers))

        # Priority 4: PHASE_LAG
        match, conf, triggers = self._check_phase_lag(metrics, window_history)
        if match:
            candidates.append((RTTSPattern.PHASE_LAG, conf, triggers))

        # Priority 5: DRIFT
        match, conf, triggers = self._check_drift(metrics, window_history)
        if match:
            candidates.append((RTTSPattern.DRIFT, conf, triggers))

        # Priority 6: TRANSIENT_MISS
        match, conf, triggers = self._check_transient_miss(metrics, window_history)
        if match:
            candidates.append((RTTSPattern.TRANSIENT_MISS, conf, triggers))

        # Step 4: Select best match (first with confidence >= threshold)
        for pattern, conf, triggers in candidates:
            if conf >= self.CONFIDENCE_THRESHOLD:
                return PatternClassification(
                    pattern=pattern,
                    confidence=conf,
                    primary_triggers=triggers.get("primary", []),
                    secondary_triggers=triggers.get("secondary", []),
                    metrics_snapshot=metrics,
                    window_id=window_id,
                    cycle_range=cycle_range,
                )

        # No pattern detected - stable state
        return PatternClassification(
            pattern=RTTSPattern.NONE,
            confidence=1.0,
            primary_triggers=[],
            secondary_triggers=["all_metrics_nominal"],
            metrics_snapshot=metrics,
            window_id=window_id,
            cycle_range=cycle_range,
        )

    def _merge_metrics(
        self,
        p3_tda: Optional[TDASummary],
        p4_tda: Optional[TDASummary],
    ) -> Dict[str, float]:
        """Merge metrics from P3 and P4 sources, preferring P4 for DRS only."""
        metrics: Dict[str, float] = {}

        # Extract from P3 first (baseline)
        if p3_tda:
            metrics["sns"] = p3_tda.sns_mean
            metrics["sns_max"] = p3_tda.sns_max
            metrics["pcs"] = p3_tda.pcs_mean
            metrics["pcs_min"] = p3_tda.pcs_min
            metrics["hss"] = p3_tda.hss_mean
            metrics["hss_min"] = p3_tda.hss_min
            metrics["envelope_occupancy"] = p3_tda.envelope_occupancy
            metrics["envelope_exit_total"] = float(p3_tda.envelope_exit_total)
            metrics["total_red_flags"] = float(p3_tda.total_red_flags)

        # Extract from P4 (DRS is P4-specific; other metrics only override if P3 not available)
        if p4_tda:
            metrics["drs"] = p4_tda.drs_mean
            metrics["drs_max"] = p4_tda.drs_max
            # Only use P4's SNS/PCS/HSS if P3 wasn't provided
            if p3_tda is None:
                metrics["sns"] = p4_tda.sns_mean
                metrics["sns_max"] = p4_tda.sns_max
                metrics["pcs"] = p4_tda.pcs_mean
                metrics["pcs_min"] = p4_tda.pcs_min
                metrics["hss"] = p4_tda.hss_mean
                metrics["hss_min"] = p4_tda.hss_min
                metrics["envelope_occupancy"] = p4_tda.envelope_occupancy
                metrics["envelope_exit_total"] = float(p4_tda.envelope_exit_total)
                metrics["total_red_flags"] = float(p4_tda.total_red_flags)

        # Defaults for missing metrics
        metrics.setdefault("sns", 0.0)
        metrics.setdefault("sns_max", 0.0)
        metrics.setdefault("pcs", 1.0)
        metrics.setdefault("pcs_min", 1.0)
        metrics.setdefault("hss", 1.0)
        metrics.setdefault("hss_min", 1.0)
        metrics.setdefault("drs", 0.0)
        metrics.setdefault("drs_max", 0.0)
        metrics.setdefault("envelope_occupancy", 1.0)
        metrics.setdefault("envelope_exit_total", 0.0)
        metrics.setdefault("total_red_flags", 0.0)

        return metrics

    def _compute_deltas(
        self,
        window_history: List[TDAWindowMetrics],
    ) -> Dict[str, float]:
        """Compute delta/trend metrics from window history."""
        deltas: Dict[str, float] = {}

        if len(window_history) < 2:
            return deltas

        # Get last two windows
        prev = window_history[-2]
        curr = window_history[-1]

        # Compute deltas
        deltas["hss_delta"] = curr.hss_mean - prev.hss_mean
        deltas["pcs_delta"] = curr.pcs_mean - prev.pcs_mean
        deltas["sns_delta"] = curr.sns_mean - prev.sns_mean

        # Compute variance if we have enough windows
        if len(window_history) >= 3:
            sns_values = [w.sns_mean for w in window_history[-5:]]  # Last 5 windows
            if len(sns_values) >= 2:
                mean_sns = sum(sns_values) / len(sns_values)
                deltas["sns_variance"] = sum((x - mean_sns) ** 2 for x in sns_values) / len(sns_values)
            else:
                deltas["sns_variance"] = 0.0

        # Compute DRS slope if available
        if len(window_history) >= 3:
            # Simple linear regression slope for DRS
            # Note: TDAWindowMetrics doesn't have drs_mean, so we skip if not available
            deltas["drs_slope"] = 0.0

        # Track HSS slope (is it decreasing?)
        if len(window_history) >= 3:
            hss_values = [w.hss_mean for w in window_history[-3:]]
            if hss_values[0] > 0:
                deltas["hss_slope"] = (hss_values[-1] - hss_values[0]) / len(hss_values)
            else:
                deltas["hss_slope"] = 0.0

        return deltas

    def _check_drift(
        self,
        metrics: Dict[str, float],
        history: Optional[List[TDAWindowMetrics]],
    ) -> Tuple[bool, float, Dict[str, List[str]]]:
        """
        Check DRIFT conditions.

        DRIFT := (DRS > 0.05 for 3+ cycles OR DRS_slope > 0.01/cycle)
                 AND SNS < 0.4
                 AND HSS > 0.6
        """
        primary_triggers: List[str] = []
        secondary_triggers: List[str] = []

        drs = metrics.get("drs", 0.0)
        sns = metrics.get("sns", 0.0)
        hss = metrics.get("hss", 1.0)
        pcs = metrics.get("pcs", 1.0)
        drs_slope = metrics.get("drs_slope", 0.0)

        # Primary: DRS elevated
        primary_match = False
        primary_strength = 0.0

        if drs > self.DRIFT_DRS_THRESHOLD:
            primary_triggers.append(f"drs_above_{self.DRIFT_DRS_THRESHOLD}")
            primary_match = True
            # Strength: how far DRS exceeds threshold
            strength = min(1.0, (drs - self.DRIFT_DRS_THRESHOLD) / self.DRIFT_DRS_THRESHOLD)
            primary_strength = max(0.5, strength)  # Minimum 0.5 when triggered

        if drs_slope > self.DRIFT_DRS_SLOPE_THRESHOLD:
            primary_triggers.append(f"drs_slope_above_{self.DRIFT_DRS_SLOPE_THRESHOLD}")
            primary_match = True
            slope_strength = min(1.0, drs_slope / self.DRIFT_DRS_SLOPE_THRESHOLD)
            slope_strength = max(0.5, slope_strength)
            primary_strength = max(primary_strength, slope_strength)

        if not primary_match:
            return False, 0.0, {"primary": [], "secondary": []}

        # Secondary conditions
        secondary_count = 0
        secondary_total = 3

        if sns < self.DRIFT_SNS_CEILING:
            secondary_triggers.append(f"sns_below_{self.DRIFT_SNS_CEILING}")
            secondary_count += 1

        if hss > self.DRIFT_HSS_FLOOR:
            secondary_triggers.append(f"hss_above_{self.DRIFT_HSS_FLOOR}")
            secondary_count += 1

        if pcs > self.DRIFT_PCS_FLOOR:
            secondary_triggers.append(f"pcs_above_{self.DRIFT_PCS_FLOOR}")
            secondary_count += 1

        # Need at least 1 secondary
        if secondary_count == 0:
            return False, 0.0, {"primary": primary_triggers, "secondary": []}

        confidence = self._compute_confidence(primary_strength, secondary_count, secondary_total)

        return True, confidence, {"primary": primary_triggers, "secondary": secondary_triggers}

    def _check_phase_lag(
        self,
        metrics: Dict[str, float],
        history: Optional[List[TDAWindowMetrics]],
    ) -> Tuple[bool, float, Dict[str, List[str]]]:
        """
        Check PHASE_LAG conditions.

        PHASE_LAG := (PCS < 0.5 OR PCS_delta < -0.15)
                     AND 0.03 < DRS < 0.10
                     AND HSS > 0.5
        """
        primary_triggers: List[str] = []
        secondary_triggers: List[str] = []

        pcs = metrics.get("pcs", 1.0)
        pcs_delta = metrics.get("pcs_delta", 0.0)
        drs = metrics.get("drs", 0.0)
        hss = metrics.get("hss", 1.0)
        sns = metrics.get("sns", 0.0)

        # Primary: PCS breakdown
        primary_match = False
        primary_strength = 0.0

        if pcs < self.PHASE_LAG_PCS_THRESHOLD:
            primary_triggers.append(f"pcs_below_{self.PHASE_LAG_PCS_THRESHOLD}")
            primary_match = True
            primary_strength = min(1.0, (self.PHASE_LAG_PCS_THRESHOLD - pcs) / self.PHASE_LAG_PCS_THRESHOLD)

        if pcs_delta < self.PHASE_LAG_PCS_DELTA_THRESHOLD:
            primary_triggers.append(f"pcs_delta_below_{self.PHASE_LAG_PCS_DELTA_THRESHOLD}")
            primary_match = True
            delta_strength = min(1.0, abs(pcs_delta - self.PHASE_LAG_PCS_DELTA_THRESHOLD) / abs(self.PHASE_LAG_PCS_DELTA_THRESHOLD))
            primary_strength = max(primary_strength, delta_strength)

        if not primary_match:
            return False, 0.0, {"primary": [], "secondary": []}

        # Secondary conditions
        secondary_count = 0
        secondary_total = 3

        if self.PHASE_LAG_DRS_MIN < drs < self.PHASE_LAG_DRS_MAX:
            secondary_triggers.append(f"drs_in_range_{self.PHASE_LAG_DRS_MIN}_{self.PHASE_LAG_DRS_MAX}")
            secondary_count += 1

        if hss > self.PHASE_LAG_HSS_FLOOR:
            secondary_triggers.append(f"hss_above_{self.PHASE_LAG_HSS_FLOOR}")
            secondary_count += 1

        if sns < self.PHASE_LAG_SNS_CEILING:
            secondary_triggers.append(f"sns_below_{self.PHASE_LAG_SNS_CEILING}")
            secondary_count += 1

        if secondary_count == 0:
            return False, 0.0, {"primary": primary_triggers, "secondary": []}

        confidence = self._compute_confidence(primary_strength, secondary_count, secondary_total)

        return True, confidence, {"primary": primary_triggers, "secondary": secondary_triggers}

    def _check_structural_break(
        self,
        metrics: Dict[str, float],
        history: Optional[List[TDAWindowMetrics]],
    ) -> Tuple[bool, float, Dict[str, List[str]]]:
        """
        Check STRUCTURAL_BREAK conditions.

        STRUCTURAL_BREAK := (HSS < 0.5 OR HSS_delta < -0.25)
                            AND SNS > 0.5
        """
        primary_triggers: List[str] = []
        secondary_triggers: List[str] = []

        hss = metrics.get("hss", 1.0)
        hss_delta = metrics.get("hss_delta", 0.0)
        sns = metrics.get("sns", 0.0)
        drs = metrics.get("drs", 0.0)

        # Primary: HSS collapse
        primary_match = False
        primary_strength = 0.0

        if hss < self.STRUCTURAL_BREAK_HSS_THRESHOLD:
            primary_triggers.append(f"hss_below_{self.STRUCTURAL_BREAK_HSS_THRESHOLD}")
            primary_match = True
            # Strength: how far below threshold (0.5). At 0, strength = 1.0; at 0.5, strength = 0
            primary_strength = min(1.0, (self.STRUCTURAL_BREAK_HSS_THRESHOLD - hss) / self.STRUCTURAL_BREAK_HSS_THRESHOLD)
            # Ensure minimum strength of 0.5 when condition is met
            primary_strength = max(0.5, primary_strength)

        if hss_delta < self.STRUCTURAL_BREAK_HSS_DELTA_THRESHOLD:
            primary_triggers.append(f"hss_delta_below_{self.STRUCTURAL_BREAK_HSS_DELTA_THRESHOLD}")
            primary_match = True
            # Strength: how far delta exceeds threshold
            delta_strength = min(1.0, abs(hss_delta) / abs(self.STRUCTURAL_BREAK_HSS_DELTA_THRESHOLD))
            delta_strength = max(0.5, delta_strength)  # Minimum 0.5 when triggered
            primary_strength = max(primary_strength, delta_strength)

        if not primary_match:
            return False, 0.0, {"primary": [], "secondary": []}

        # Secondary: SNS spike (required)
        secondary_count = 0
        secondary_total = 2

        if sns > self.STRUCTURAL_BREAK_SNS_THRESHOLD:
            secondary_triggers.append(f"sns_above_{self.STRUCTURAL_BREAK_SNS_THRESHOLD}")
            secondary_count += 1
        else:
            # SNS is required for STRUCTURAL_BREAK
            return False, 0.0, {"primary": primary_triggers, "secondary": []}

        if drs > self.STRUCTURAL_BREAK_DRS_THRESHOLD:
            secondary_triggers.append(f"drs_above_{self.STRUCTURAL_BREAK_DRS_THRESHOLD}")
            secondary_count += 1

        confidence = self._compute_confidence(primary_strength, secondary_count, secondary_total)

        return True, confidence, {"primary": primary_triggers, "secondary": secondary_triggers}

    def _check_noise_amplification(
        self,
        metrics: Dict[str, float],
        history: Optional[List[TDAWindowMetrics]],
    ) -> Tuple[bool, float, Dict[str, List[str]]]:
        """
        Check NOISE_AMPLIFICATION conditions.

        NOISE_AMPLIFICATION := (SNS_variance > 0.04 OR (SNS_mean > 0.35 AND SNS_max > 0.6))
                               AND PCS < 0.6
                               AND HSS > 0.4
        """
        primary_triggers: List[str] = []
        secondary_triggers: List[str] = []

        sns = metrics.get("sns", 0.0)
        sns_max = metrics.get("sns_max", 0.0)
        sns_variance = metrics.get("sns_variance", 0.0)
        pcs = metrics.get("pcs", 1.0)
        hss = metrics.get("hss", 1.0)
        envelope_exit_total = metrics.get("envelope_exit_total", 0.0)

        # Primary: SNS instability
        primary_match = False
        primary_strength = 0.0

        if sns_variance > self.NOISE_AMP_SNS_VARIANCE_THRESHOLD:
            primary_triggers.append(f"sns_variance_above_{self.NOISE_AMP_SNS_VARIANCE_THRESHOLD}")
            primary_match = True
            # Strength based on how much variance exceeds threshold
            primary_strength = min(1.0, sns_variance / self.NOISE_AMP_SNS_VARIANCE_THRESHOLD)

        if sns >= self.NOISE_AMP_SNS_MEAN_THRESHOLD and sns_max >= self.NOISE_AMP_SNS_MAX_THRESHOLD:
            primary_triggers.append(f"sns_elevated_with_spikes")
            primary_match = True
            # Strength based on how elevated SNS is - use mean and max
            mean_strength = min(1.0, (sns - self.NOISE_AMP_SNS_MEAN_THRESHOLD) / (1.0 - self.NOISE_AMP_SNS_MEAN_THRESHOLD))
            max_strength = min(1.0, (sns_max - self.NOISE_AMP_SNS_MAX_THRESHOLD) / (1.0 - self.NOISE_AMP_SNS_MAX_THRESHOLD))
            spike_strength = 0.5 + 0.5 * max(mean_strength, max_strength)  # Base 0.5 + bonus
            primary_strength = max(primary_strength, spike_strength)

        if not primary_match:
            return False, 0.0, {"primary": [], "secondary": []}

        # Secondary conditions
        secondary_count = 0
        secondary_total = 3

        if pcs < self.NOISE_AMP_PCS_CEILING:
            secondary_triggers.append(f"pcs_below_{self.NOISE_AMP_PCS_CEILING}")
            secondary_count += 1

        if hss > self.NOISE_AMP_HSS_FLOOR:
            secondary_triggers.append(f"hss_above_{self.NOISE_AMP_HSS_FLOOR}")
            secondary_count += 1

        if envelope_exit_total > self.NOISE_AMP_ENVELOPE_EXITS_THRESHOLD:
            secondary_triggers.append(f"envelope_exits_above_{self.NOISE_AMP_ENVELOPE_EXITS_THRESHOLD}")
            secondary_count += 1

        if secondary_count == 0:
            return False, 0.0, {"primary": primary_triggers, "secondary": []}

        confidence = self._compute_confidence(primary_strength, secondary_count, secondary_total)

        return True, confidence, {"primary": primary_triggers, "secondary": secondary_triggers}

    def _check_attractor_miss(
        self,
        metrics: Dict[str, float],
        history: Optional[List[TDAWindowMetrics]],
    ) -> Tuple[bool, float, Dict[str, List[str]]]:
        """
        Check ATTRACTOR_MISS conditions.

        ATTRACTOR_MISS := HSS < 0.6 AND HSS_slope < 0
                          AND DRS > 0.10
                          AND 0.3 < SNS < 0.6
        """
        primary_triggers: List[str] = []
        secondary_triggers: List[str] = []

        hss = metrics.get("hss", 1.0)
        hss_slope = metrics.get("hss_slope", 0.0)
        drs = metrics.get("drs", 0.0)
        sns = metrics.get("sns", 0.0)
        pcs = metrics.get("pcs", 1.0)

        # Primary: HSS degrading + DRS high
        primary_match = False
        primary_strength = 0.0

        hss_condition = hss < self.ATTRACTOR_MISS_HSS_THRESHOLD and hss_slope < 0
        drs_condition = drs > self.ATTRACTOR_MISS_DRS_THRESHOLD

        if hss_condition:
            primary_triggers.append(f"hss_below_{self.ATTRACTOR_MISS_HSS_THRESHOLD}_decreasing")
            hss_strength = min(1.0, (self.ATTRACTOR_MISS_HSS_THRESHOLD - hss) / self.ATTRACTOR_MISS_HSS_THRESHOLD)
            primary_strength = hss_strength

        if drs_condition:
            primary_triggers.append(f"drs_above_{self.ATTRACTOR_MISS_DRS_THRESHOLD}")
            drs_strength = min(1.0, (drs - self.ATTRACTOR_MISS_DRS_THRESHOLD) / self.ATTRACTOR_MISS_DRS_THRESHOLD)
            primary_strength = max(primary_strength, drs_strength)

        # Both conditions required for primary match
        if not (hss_condition and drs_condition):
            return False, 0.0, {"primary": [], "secondary": []}

        primary_match = True

        # Secondary conditions
        secondary_count = 0
        secondary_total = 2

        if self.ATTRACTOR_MISS_SNS_MIN < sns < self.ATTRACTOR_MISS_SNS_MAX:
            secondary_triggers.append(f"sns_in_range_{self.ATTRACTOR_MISS_SNS_MIN}_{self.ATTRACTOR_MISS_SNS_MAX}")
            secondary_count += 1

        if pcs < self.ATTRACTOR_MISS_PCS_CEILING:
            secondary_triggers.append(f"pcs_below_{self.ATTRACTOR_MISS_PCS_CEILING}")
            secondary_count += 1

        # At least one secondary needed
        if secondary_count == 0:
            return False, 0.0, {"primary": primary_triggers, "secondary": []}

        confidence = self._compute_confidence(primary_strength, secondary_count, secondary_total)

        return True, confidence, {"primary": primary_triggers, "secondary": secondary_triggers}

    def _check_transient_miss(
        self,
        metrics: Dict[str, float],
        history: Optional[List[TDAWindowMetrics]],
    ) -> Tuple[bool, float, Dict[str, List[str]]]:
        """
        Check TRANSIENT_MISS conditions.

        TRANSIENT_MISS := SNS_peak > 0.5 AND SNS_current < 0.3
                          AND HSS > 0.6
                          AND DRS < 0.08
                          AND cycles_since_peak <= 3
        """
        primary_triggers: List[str] = []
        secondary_triggers: List[str] = []

        sns = metrics.get("sns", 0.0)
        sns_max = metrics.get("sns_max", 0.0)
        hss = metrics.get("hss", 1.0)
        drs = metrics.get("drs", 0.0)

        # Primary: SNS spike followed by recovery
        primary_match = False
        primary_strength = 0.0

        # Check if we had a spike (sns_max > threshold) and recovered (current sns < recovery)
        had_spike = sns_max > self.TRANSIENT_MISS_SNS_PEAK_THRESHOLD
        recovered = sns < self.TRANSIENT_MISS_SNS_RECOVERY_THRESHOLD

        if had_spike and recovered:
            primary_triggers.append(f"sns_spike_recovered")
            primary_match = True
            # Strength based on how clear the spike/recovery is
            spike_magnitude = sns_max - self.TRANSIENT_MISS_SNS_PEAK_THRESHOLD
            recovery_depth = self.TRANSIENT_MISS_SNS_RECOVERY_THRESHOLD - sns
            primary_strength = min(1.0, (spike_magnitude + max(0, recovery_depth)) / 0.5)

        if not primary_match:
            return False, 0.0, {"primary": [], "secondary": []}

        # Secondary conditions
        secondary_count = 0
        secondary_total = 2

        if hss > self.TRANSIENT_MISS_HSS_FLOOR:
            secondary_triggers.append(f"hss_above_{self.TRANSIENT_MISS_HSS_FLOOR}")
            secondary_count += 1

        if drs < self.TRANSIENT_MISS_DRS_CEILING:
            secondary_triggers.append(f"drs_below_{self.TRANSIENT_MISS_DRS_CEILING}")
            secondary_count += 1

        if secondary_count == 0:
            return False, 0.0, {"primary": primary_triggers, "secondary": []}

        confidence = self._compute_confidence(primary_strength, secondary_count, secondary_total)

        return True, confidence, {"primary": primary_triggers, "secondary": secondary_triggers}

    def _compute_confidence(
        self,
        primary_match_strength: float,
        secondary_match_count: int,
        secondary_total: int,
    ) -> float:
        """
        Compute confidence score for a pattern match.

        Formula:
            confidence = 0.6 * primary_match_strength
                       + 0.4 * (secondary_match_count / secondary_total)

        Returns value in [0.0, 1.0].
        """
        if secondary_total == 0:
            secondary_ratio = 1.0
        else:
            secondary_ratio = secondary_match_count / secondary_total

        confidence = 0.6 * primary_match_strength + 0.4 * secondary_ratio

        return max(0.0, min(1.0, confidence))


def attach_tda_patterns_to_evidence(
    evidence: Dict[str, Any],
    classifier: TDAPatternClassifier,
    p3_tda: Optional[TDASummary] = None,
    p4_tda: Optional[TDASummary] = None,
    window_history: Optional[List[TDAWindowMetrics]] = None,
    classification_history: Optional[List[PatternClassification]] = None,
    window_id: Optional[str] = None,
    cycle_range: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """
    Attach TDA pattern classification to evidence pack.

    Adds governance.tda.patterns section with:
      - Current classification
      - Classification history (last N windows)
      - Pattern summary statistics

    SHADOW MODE: Observational only.

    Args:
        evidence: Evidence pack dict to augment
        classifier: TDAPatternClassifier instance
        p3_tda: P3 summary for classification
        p4_tda: P4 summary for classification
        window_history: Recent windows for trend analysis
        classification_history: Previous classifications for history section
        window_id: Optional window identifier
        cycle_range: Optional (start_cycle, end_cycle) tuple

    Returns:
        Modified evidence dict with patterns section added
    """
    # Ensure governance.tda structure exists
    if "governance" not in evidence:
        evidence["governance"] = {}
    if "tda" not in evidence["governance"]:
        evidence["governance"]["tda"] = {"mode": "SHADOW"}

    # Perform current classification
    current_classification = classifier.classify(
        p3_tda=p3_tda,
        p4_tda=p4_tda,
        window_history=window_history,
        window_id=window_id,
        cycle_range=cycle_range,
    )

    # Build classification history for evidence
    history_for_evidence: List[Dict[str, Any]] = []
    if classification_history:
        for clf in classification_history[-10:]:  # Last 10 classifications
            history_for_evidence.append({
                "window_id": clf.window_id,
                "pattern": clf.pattern.value,
                "confidence": round(clf.confidence, 4),
            })

    # Add current to history
    history_for_evidence.append({
        "window_id": current_classification.window_id,
        "pattern": current_classification.pattern.value,
        "confidence": round(current_classification.confidence, 4),
    })

    # Compute pattern counts for summary
    all_classifications = (classification_history or []) + [current_classification]
    pattern_counts: Dict[str, int] = {}
    high_confidence_events = 0

    for clf in all_classifications:
        pattern_name = clf.pattern.value
        pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
        if clf.confidence >= 0.75 and clf.pattern != RTTSPattern.NONE:
            high_confidence_events += 1

    # Determine dominant pattern (excluding NONE)
    non_none_counts = {k: v for k, v in pattern_counts.items() if k != "NONE"}
    if non_none_counts:
        dominant_pattern = max(non_none_counts, key=lambda k: non_none_counts[k])
    else:
        dominant_pattern = "NONE"

    # Build patterns section
    patterns_section = {
        "schema_version": "1.0.0",
        "classifier_version": "TDAPatternClassifier-v1",
        "classification": current_classification.to_dict(),
        "classification_history": history_for_evidence,
        "summary": {
            "dominant_pattern": dominant_pattern,
            "pattern_counts": pattern_counts,
            "high_confidence_events": high_confidence_events,
            "total_windows_classified": len(all_classifications),
        },
        "verifier_note": "Pattern classification is SHADOW-ONLY. No enforcement triggered.",
        "mode": "SHADOW",
    }

    evidence["governance"]["tda"]["patterns"] = patterns_section

    return evidence
