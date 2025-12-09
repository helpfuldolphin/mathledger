"""
Curriculum Stability Integration

Integrates the stability envelope with curriculum gates to prevent
invalid slice transitions during integration.

This module bridges the stability envelope (TDA-driven HSS tracking)
with the curriculum gate system (coverage, abstention, velocity, caps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from curriculum.gates import (
    GateStatus,
    GateVerdict,
    NormalizedMetrics,
    CurriculumSlice,
)
from curriculum.stability_envelope import (
    CurriculumStabilityEnvelope,
    StabilityEnvelopeConfig,
)


@dataclass
class StabilityGateSpec:
    """Specification for stability gate."""
    
    enabled: bool = True
    min_suitability_score: float = 0.6
    allow_variance_spikes: bool = False
    require_stable_slice: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_suitability_score": self.min_suitability_score,
            "allow_variance_spikes": self.allow_variance_spikes,
            "require_stable_slice": self.require_stable_slice,
        }


class StabilityGateEvaluator:
    """
    Evaluates stability as a curriculum gate.
    
    This integrates with the existing gate system (coverage, abstention,
    velocity, caps) to add stability checks before allowing slice transitions.
    """
    
    def __init__(
        self,
        envelope: CurriculumStabilityEnvelope,
        spec: StabilityGateSpec,
        slice_cfg: CurriculumSlice,
    ):
        """
        Initialize stability gate evaluator.
        
        Args:
            envelope: Stability envelope tracker
            spec: Stability gate specification
            slice_cfg: Current curriculum slice configuration
        """
        self.envelope = envelope
        self.spec = spec
        self.slice = slice_cfg
    
    def evaluate(self) -> GateStatus:
        """
        Evaluate stability gate for current slice.
        
        Returns:
            GateStatus indicating whether stability requirements are met
        """
        if not self.spec.enabled:
            return GateStatus(
                gate="stability",
                passed=True,
                observed={"enabled": False},
                thresholds=self.spec.to_dict(),
                message="stability gate disabled",
            )
        
        slice_name = self.slice.name
        
        # Compute slice stability metrics
        stability_metrics = self.envelope.compute_slice_stability(slice_name)
        
        # Check for variance spike
        spike_detected, current_var = self.envelope.detect_variance_spike(slice_name)
        
        observed = {
            "slice_name": slice_name,
            "hss_mean": stability_metrics.hss_mean,
            "hss_cv": stability_metrics.hss_cv,
            "suitability_score": stability_metrics.suitability_score,
            "is_stable": stability_metrics.is_stable,
            "variance_spike_detected": spike_detected,
            "current_variance": current_var,
            "flags": stability_metrics.flags,
            "total_cycles": stability_metrics.total_cycles,
        }
        
        # Check insufficient data
        if "insufficient_data" in stability_metrics.flags:
            return GateStatus(
                gate="stability",
                passed=False,
                observed=observed,
                thresholds=self.spec.to_dict(),
                message=f"slice '{slice_name}' has insufficient HSS data",
            )
        
        # Check if slice must be stable
        if self.spec.require_stable_slice and not stability_metrics.is_stable:
            return GateStatus(
                gate="stability",
                passed=False,
                observed=observed,
                thresholds=self.spec.to_dict(),
                message=f"slice '{slice_name}' is unstable: {', '.join(stability_metrics.flags)}",
            )
        
        # Check suitability score
        if stability_metrics.suitability_score < self.spec.min_suitability_score:
            return GateStatus(
                gate="stability",
                passed=False,
                observed=observed,
                thresholds=self.spec.to_dict(),
                message=(
                    f"suitability score {stability_metrics.suitability_score:.3f} "
                    f"< {self.spec.min_suitability_score:.3f}"
                ),
            )
        
        # Check variance spike
        if not self.spec.allow_variance_spikes and spike_detected:
            return GateStatus(
                gate="stability",
                passed=False,
                observed=observed,
                thresholds=self.spec.to_dict(),
                message=f"HSS variance spike detected in slice '{slice_name}'",
            )
        
        # All checks passed
        return GateStatus(
            gate="stability",
            passed=True,
            observed=observed,
            thresholds=self.spec.to_dict(),
            message=(
                f"slice '{slice_name}' is stable: "
                f"suitability={stability_metrics.suitability_score:.3f}, "
                f"cv={stability_metrics.hss_cv:.4f}"
            ),
        )


def should_ratchet_with_stability(
    metrics: Dict[str, Any],
    slice_cfg: CurriculumSlice,
    envelope: CurriculumStabilityEnvelope,
    stability_spec: Optional[StabilityGateSpec] = None,
) -> GateVerdict:
    """
    Enhanced ratchet decision including stability gate.
    
    This extends the standard gate evaluation (coverage, abstention,
    velocity, caps) with stability checks (HSS variance, suitability).
    
    Args:
        metrics: Raw metrics from experiment run
        slice_cfg: Current curriculum slice configuration
        envelope: Stability envelope tracker
        stability_spec: Stability gate specification (uses default if None)
        
    Returns:
        GateVerdict with stability gate included in evaluation
    """
    # Use default stability spec if not provided
    if stability_spec is None:
        stability_spec = StabilityGateSpec()
    
    # Evaluate stability gate
    stability_evaluator = StabilityGateEvaluator(envelope, stability_spec, slice_cfg)
    stability_status = stability_evaluator.evaluate()
    
    # Build audit record
    audit = {
        "slice": slice_cfg.name,
        "stability_gate": stability_status.to_dict(),
    }
    
    # Check if stability gate failed
    if not stability_status.passed:
        audit["summary"] = stability_status.message
        return GateVerdict(
            advance=False,
            reason=f"stability gate: {stability_status.message}",
            audit=audit,
        )
    
    # Stability gate passed
    audit["summary"] = stability_status.message
    return GateVerdict(
        advance=True,
        reason=stability_status.message,
        audit=audit,
    )


def record_cycle_hss_metrics(
    envelope: CurriculumStabilityEnvelope,
    cycle_id: str,
    slice_name: str,
    metrics: Dict[str, Any],
    timestamp: str,
) -> None:
    """
    Record HSS metrics from a cycle into the stability envelope.
    
    This is a convenience function to extract HSS-related metrics from
    the raw metrics dict and record them in the envelope.
    
    Args:
        envelope: Stability envelope tracker
        cycle_id: Unique cycle identifier
        slice_name: Current slice name
        metrics: Raw metrics from cycle
        timestamp: ISO timestamp
    """
    # Extract HSS value from metrics
    # HSS can come from multiple sources in the metrics dict
    hss_value = _extract_hss_value(metrics)
    
    # Extract verified count
    verified_count = _extract_verified_count(metrics)
    
    # Record in envelope
    envelope.record_cycle(
        cycle_id=cycle_id,
        slice_name=slice_name,
        hss_value=hss_value,
        verified_count=verified_count,
        timestamp=timestamp,
    )


def _extract_hss_value(metrics: Dict[str, Any]) -> float:
    """
    Extract HSS value from metrics dictionary.
    
    HSS (Homological Spanning Set) score represents topological diversity.
    Falls back to coverage rate if HSS not available.
    """
    root = metrics.get("metrics", metrics)
    
    # Try to find HSS value in various locations
    # Priority: tda.hss > topology.hss > coverage (as proxy)
    if "tda" in root and isinstance(root["tda"], dict):
        if "hss" in root["tda"]:
            return float(root["tda"]["hss"])
        if "hss_score" in root["tda"]:
            return float(root["tda"]["hss_score"])
    
    if "topology" in root and isinstance(root["topology"], dict):
        if "hss" in root["topology"]:
            return float(root["topology"]["hss"])
        if "diversity" in root["topology"]:
            return float(root["topology"]["diversity"])
    
    # Fallback: use coverage as HSS proxy
    # Coverage roughly correlates with topological diversity
    if "rfl" in root and isinstance(root["rfl"], dict):
        if "coverage" in root["rfl"] and isinstance(root["rfl"]["coverage"], dict):
            ci_lower = root["rfl"]["coverage"].get("ci_lower")
            if ci_lower is not None:
                return float(ci_lower)
    
    if "coverage" in root:
        if isinstance(root["coverage"], dict):
            ci_lower = root["coverage"].get("ci_lower")
            if ci_lower is not None:
                return float(ci_lower)
        elif isinstance(root["coverage"], (int, float)):
            return float(root["coverage"])
    
    # Default: assume moderate HSS if no data available
    return 0.5


def _extract_verified_count(metrics: Dict[str, Any]) -> int:
    """Extract verified proof count from metrics."""
    root = metrics.get("metrics", metrics)
    
    # Try various locations for verified count
    if "proofs" in root and isinstance(root["proofs"], dict):
        verified = root["proofs"].get("verified")
        if verified is not None:
            return int(verified)
        
        successful = root["proofs"].get("successful")
        if successful is not None:
            return int(successful)
    
    if "verified_count" in root:
        return int(root["verified_count"])
    
    if "successful_proofs" in root:
        return int(root["successful_proofs"])
    
    # Default: 0 if not found
    return 0
