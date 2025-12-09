"""
Curriculum Stability Envelope

Detects TDA-driven slice instability and provides slice suitability scoring
for uplift evaluation. Prevents invalid curriculum changes during integration
by tracking HSS variance and flagging unstable slices.

Integration Point: Called by curriculum gates before slice transitions.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class HSSMetrics:
    """Homological Spanning Set (HSS) metrics for a single cycle/run."""
    
    cycle_id: str
    hss_value: float  # HSS score for this cycle
    verified_count: int  # Number of verified proofs
    timestamp: str
    slice_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "hss_value": self.hss_value,
            "verified_count": self.verified_count,
            "timestamp": self.timestamp,
            "slice_name": self.slice_name,
        }


@dataclass
class SliceStabilityMetrics:
    """Stability metrics for a curriculum slice."""
    
    slice_name: str
    hss_mean: float
    hss_std: float
    hss_cv: float  # Coefficient of variation (std/mean)
    low_hss_count: int  # Number of cycles below threshold
    total_cycles: int
    suitability_score: float  # 0.0 (unsuitable) to 1.0 (highly suitable)
    is_stable: bool
    flags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "slice_name": self.slice_name,
            "hss_mean": self.hss_mean,
            "hss_std": self.hss_std,
            "hss_cv": self.hss_cv,
            "low_hss_count": self.low_hss_count,
            "total_cycles": self.total_cycles,
            "suitability_score": self.suitability_score,
            "is_stable": self.is_stable,
            "flags": self.flags,
        }


@dataclass
class StabilityEnvelopeConfig:
    """Configuration for stability envelope detection."""
    
    # HSS variance thresholds
    max_hss_cv: float = 0.25  # Max coefficient of variation for stability
    min_hss_threshold: float = 0.3  # Minimum HSS value considered "low"
    max_low_hss_ratio: float = 0.2  # Max ratio of low-HSS cycles allowed
    
    # Suitability scoring weights
    weight_mean: float = 0.4  # Weight for mean HSS value
    weight_stability: float = 0.3  # Weight for stability (low CV)
    weight_consistency: float = 0.3  # Weight for consistency (few low-HSS)
    
    # Minimum samples for stability evaluation
    min_cycles_for_stability: int = 5
    
    # Variance spike detection
    variance_spike_threshold: float = 2.0  # Multiplier over baseline variance
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_hss_cv": self.max_hss_cv,
            "min_hss_threshold": self.min_hss_threshold,
            "max_low_hss_ratio": self.max_low_hss_ratio,
            "weight_mean": self.weight_mean,
            "weight_stability": self.weight_stability,
            "weight_consistency": self.weight_consistency,
            "min_cycles_for_stability": self.min_cycles_for_stability,
            "variance_spike_threshold": self.variance_spike_threshold,
        }


class CurriculumStabilityEnvelope:
    """
    Tracks curriculum slice stability using TDA-driven HSS metrics.
    
    Detects:
    - HSS variance spikes indicating slice instability
    - Repeated low-HSS regions suggesting unsuitable slices
    - Overall slice suitability for uplift experiments
    
    Integration: Called by curriculum gates before allowing slice transitions.
    """
    
    def __init__(self, config: Optional[StabilityEnvelopeConfig] = None):
        """
        Initialize stability envelope tracker.
        
        Args:
            config: Configuration for stability detection thresholds
        """
        self.config = config or StabilityEnvelopeConfig()
        self.hss_history: Dict[str, List[HSSMetrics]] = {}  # slice_name -> metrics
        self.baseline_variance: Dict[str, float] = {}  # slice_name -> baseline variance
    
    def record_cycle(
        self,
        cycle_id: str,
        slice_name: str,
        hss_value: float,
        verified_count: int,
        timestamp: str
    ) -> None:
        """
        Record HSS metrics for a single cycle.
        
        Args:
            cycle_id: Unique cycle identifier
            slice_name: Curriculum slice name
            hss_value: HSS score for this cycle (0.0 to 1.0)
            verified_count: Number of verified proofs
            timestamp: ISO timestamp
        """
        metrics = HSSMetrics(
            cycle_id=cycle_id,
            hss_value=hss_value,
            verified_count=verified_count,
            timestamp=timestamp,
            slice_name=slice_name,
        )
        
        if slice_name not in self.hss_history:
            self.hss_history[slice_name] = []
        
        self.hss_history[slice_name].append(metrics)
    
    def compute_slice_stability(self, slice_name: str) -> SliceStabilityMetrics:
        """
        Compute stability metrics for a curriculum slice.
        
        Args:
            slice_name: Name of the slice to evaluate
            
        Returns:
            SliceStabilityMetrics with stability assessment
        """
        if slice_name not in self.hss_history or not self.hss_history[slice_name]:
            # No data - return unstable default
            return SliceStabilityMetrics(
                slice_name=slice_name,
                hss_mean=0.0,
                hss_std=0.0,
                hss_cv=float('inf'),
                low_hss_count=0,
                total_cycles=0,
                suitability_score=0.0,
                is_stable=False,
                flags=["insufficient_data"],
            )
        
        history = self.hss_history[slice_name]
        hss_values = [m.hss_value for m in history]
        
        # Compute basic statistics
        hss_mean = statistics.mean(hss_values)
        hss_std = statistics.stdev(hss_values) if len(hss_values) > 1 else 0.0
        # CV = std/mean. If mean is 0, CV is undefined (use inf as sentinel)
        hss_cv = hss_std / hss_mean if hss_mean > 0 else float('inf')
        
        # Count low-HSS cycles
        low_hss_count = sum(1 for v in hss_values if v < self.config.min_hss_threshold)
        total_cycles = len(hss_values)
        low_hss_ratio = low_hss_count / total_cycles if total_cycles > 0 else 0.0
        
        # Determine stability flags
        flags: List[str] = []
        is_stable = True
        
        if total_cycles < self.config.min_cycles_for_stability:
            flags.append("insufficient_cycles")
            is_stable = False
        
        if hss_cv > self.config.max_hss_cv:
            flags.append("high_variance")
            is_stable = False
        
        if low_hss_ratio > self.config.max_low_hss_ratio:
            flags.append("repeated_low_hss")
            is_stable = False
        
        # Compute suitability score (0.0 to 1.0)
        suitability_score = self._compute_suitability_score(
            hss_mean, hss_cv, low_hss_ratio
        )
        
        return SliceStabilityMetrics(
            slice_name=slice_name,
            hss_mean=hss_mean,
            hss_std=hss_std,
            hss_cv=hss_cv,
            low_hss_count=low_hss_count,
            total_cycles=total_cycles,
            suitability_score=suitability_score,
            is_stable=is_stable,
            flags=flags,
        )
    
    def _compute_suitability_score(
        self,
        hss_mean: float,
        hss_cv: float,
        low_hss_ratio: float
    ) -> float:
        """
        Compute slice suitability score for uplift evaluation.
        
        Score components:
        - Mean HSS (higher is better)
        - Stability: 1.0 - normalized CV (lower CV is better)
        - Consistency: 1.0 - low_hss_ratio (fewer low-HSS is better)
        
        Returns:
            Suitability score from 0.0 (unsuitable) to 1.0 (highly suitable)
        """
        # Mean component: normalize to [0, 1]
        mean_component = min(1.0, max(0.0, hss_mean))
        
        # Stability component: penalize high CV
        # CV of 0.0 -> score 1.0, CV >= max_hss_cv -> score 0.0
        if math.isinf(hss_cv):
            stability_component = 0.0
        else:
            stability_component = max(0.0, 1.0 - (hss_cv / self.config.max_hss_cv))
        
        # Consistency component: penalize low-HSS ratio
        # 0 low-HSS -> score 1.0, ratio >= max_low_hss_ratio -> score 0.0
        consistency_component = max(0.0, 1.0 - (low_hss_ratio / self.config.max_low_hss_ratio))
        
        # Weighted combination
        score = (
            self.config.weight_mean * mean_component +
            self.config.weight_stability * stability_component +
            self.config.weight_consistency * consistency_component
        )
        
        return min(1.0, max(0.0, score))
    
    def detect_variance_spike(
        self,
        slice_name: str,
        window_size: int = 10
    ) -> Tuple[bool, Optional[float]]:
        """
        Detect if recent HSS variance has spiked compared to baseline.
        
        Args:
            slice_name: Name of the slice to check
            window_size: Number of recent cycles to check
            
        Returns:
            Tuple of (spike_detected, current_variance)
        """
        if slice_name not in self.hss_history or not self.hss_history[slice_name]:
            return False, None
        
        history = self.hss_history[slice_name]
        
        # Need at least window_size + some baseline
        if len(history) < window_size + 5:
            return False, None
        
        # Compute baseline variance (excluding recent window)
        baseline_values = [m.hss_value for m in history[:-window_size]]
        if len(baseline_values) < 2:
            return False, None
        
        baseline_var = statistics.variance(baseline_values)
        
        # Store baseline for this slice
        self.baseline_variance[slice_name] = baseline_var
        
        # Compute recent variance
        recent_values = [m.hss_value for m in history[-window_size:]]
        if len(recent_values) < 2:
            return False, None
        
        recent_var = statistics.variance(recent_values)
        
        # Detect spike: recent variance is significantly higher than baseline
        if baseline_var > 0:
            variance_ratio = recent_var / baseline_var
            spike_detected = variance_ratio > self.config.variance_spike_threshold
            return spike_detected, recent_var
        
        return False, recent_var
    
    def check_slice_transition_allowed(
        self,
        from_slice: str,
        to_slice: str
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Check if curriculum slice transition is allowed based on stability.
        
        Args:
            from_slice: Current slice name
            to_slice: Target slice name
            
        Returns:
            Tuple of (allowed, reason, details)
        """
        # Evaluate current slice stability
        from_stability = self.compute_slice_stability(from_slice)
        
        details = {
            "from_slice": from_slice,
            "to_slice": to_slice,
            "from_stability": from_stability.to_dict(),
        }
        
        # Check for variance spike
        spike_detected, current_var = self.detect_variance_spike(from_slice)
        details["variance_spike_detected"] = spike_detected
        details["current_variance"] = current_var
        
        # Block transition if current slice is unstable
        if not from_stability.is_stable:
            reason = (
                f"Cannot transition from unstable slice '{from_slice}'. "
                f"Flags: {', '.join(from_stability.flags)}. "
                f"Suitability: {from_stability.suitability_score:.3f}"
            )
            return False, reason, details
        
        # Block transition if variance spike detected
        if spike_detected:
            reason = (
                f"HSS variance spike detected in slice '{from_slice}'. "
                f"Current variance: {current_var:.4f}, "
                f"Baseline: {self.baseline_variance.get(from_slice, 0.0):.4f}"
            )
            return False, reason, details
        
        # Transition allowed
        reason = f"Slice '{from_slice}' is stable. Transition to '{to_slice}' allowed."
        return True, reason, details
    
    def get_all_slice_suitability(self) -> Dict[str, float]:
        """
        Get suitability scores for all tracked slices.
        
        Returns:
            Dict mapping slice_name to suitability_score
        """
        scores = {}
        for slice_name in self.hss_history.keys():
            stability = self.compute_slice_stability(slice_name)
            scores[slice_name] = stability.suitability_score
        return scores
    
    def export_stability_report(self) -> Dict[str, Any]:
        """
        Export complete stability report for all slices.
        
        Returns:
            Dictionary with stability metrics for all slices
        """
        report = {
            "config": self.config.to_dict(),
            "slices": {},
        }
        
        for slice_name in self.hss_history.keys():
            stability = self.compute_slice_stability(slice_name)
            spike_detected, current_var = self.detect_variance_spike(slice_name)
            
            report["slices"][slice_name] = {
                **stability.to_dict(),
                "variance_spike": spike_detected,
                "current_variance": current_var,
                "baseline_variance": self.baseline_variance.get(slice_name),
            }
        
        return report
