"""
Curriculum Stability Envelope

Tracks curriculum health via HSS (Homogeneity-Stability Score) metrics, variance tracking,
and suitability scoring. Provides bindings for P3 First Light, P4 Calibration, Evidence Packs,
and Uplift Council advisory.

HSS is a synthetic metric combining:
- Slice parameter homogeneity (how similar are slices to each other)
- Temporal stability (how consistent are metrics over time)
- Coverage variance (how much does coverage vary within a slice)

This module is shadow-mode only: it observes and reports but does not gate execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
import json


# Default thresholds for stability assessment (configurable via YAML if needed)
DEFAULT_HSS_THRESHOLD = 0.7  # Below this is "low HSS"
DEFAULT_VARIANCE_THRESHOLD = 0.15  # Above this is "high variance"
DEFAULT_SUITABILITY_THRESHOLD = 0.6  # Below this flags a slice as marginal


@dataclass
class SliceHealthMetrics:
    """Health metrics for a single curriculum slice."""
    slice_name: str
    hss: float  # Homogeneity-Stability Score (0.0-1.0)
    variance: float  # Variance metric (0.0-1.0, lower is better)
    suitability: float  # Overall suitability score (0.0-1.0)
    coverage_rate: Optional[float] = None
    abstention_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CurriculumStabilityEnvelope:
    """
    Curriculum Stability Envelope - aggregated health metrics across all slices.
    
    This envelope provides a holistic view of curriculum health for governance systems.
    """
    mean_hss: float
    hss_variance: float
    low_hss_fraction: float  # Fraction of slices below HSS threshold
    slices_flagged: List[str]  # Slice names flagged as unstable
    suitability_scores: Dict[str, float]  # Per-slice suitability (0.0-1.0)
    status_light: str  # GREEN | YELLOW | RED
    
    # Additional context
    stable_slices: List[str] = field(default_factory=list)
    unstable_slices: List[str] = field(default_factory=list)
    hss_variance_spikes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "mean_HSS": float(self.mean_hss),
            "HSS_variance": float(self.hss_variance),
            "low_HSS_fraction": float(self.low_hss_fraction),
            "slices_flagged": list(self.slices_flagged),
            "suitability_scores": {k: float(v) for k, v in self.suitability_scores.items()},
            "status_light": str(self.status_light),
            "stable_slices": list(self.stable_slices),
            "unstable_slices": list(self.unstable_slices),
            "HSS_variance_spikes": list(self.hss_variance_spikes),
        }


def compute_hss(
    slice_metrics: Dict[str, Any],
    historical_metrics: Optional[List[Dict[str, Any]]] = None
) -> float:
    """
    Compute Homogeneity-Stability Score (HSS) for a slice.
    
    HSS combines:
    - Parameter consistency (how "normal" are the parameters)
    - Temporal stability (how consistent are metrics over time)
    - Coverage variance (lower variance = higher HSS)
    
    Args:
        slice_metrics: Current metrics for the slice
        historical_metrics: Optional list of historical metric snapshots
        
    Returns:
        HSS value between 0.0 and 1.0 (higher is better)
    """
    # Component 1: Parameter homogeneity (0.0-1.0)
    # Check if parameters are within reasonable ranges
    params = slice_metrics.get("params", {})
    atoms = params.get("atoms", 0)
    depth = params.get("depth_max", 0)
    breadth = params.get("breadth_max", 0)
    
    # Simple heuristic: normalize to expected ranges
    # atoms: 3-8, depth: 3-10, breadth: 500-3000
    atoms_norm = max(0.0, min(1.0, (atoms - 3) / 5.0)) if atoms > 0 else 0.5
    depth_norm = max(0.0, min(1.0, (depth - 3) / 7.0)) if depth > 0 else 0.5
    breadth_norm = max(0.0, min(1.0, (breadth - 500) / 2500.0)) if breadth > 0 else 0.5
    
    homogeneity = (atoms_norm + depth_norm + breadth_norm) / 3.0
    
    # Component 2: Temporal stability (0.0-1.0)
    if historical_metrics and len(historical_metrics) >= 2:
        # Measure consistency of coverage rate over time
        coverage_rates = [m.get("coverage_rate", 0.0) for m in historical_metrics]
        coverage_rates = [c for c in coverage_rates if c > 0]
        if len(coverage_rates) >= 2:
            mean_cov = sum(coverage_rates) / len(coverage_rates)
            variance = sum((c - mean_cov) ** 2 for c in coverage_rates) / len(coverage_rates)
            stability = max(0.0, 1.0 - variance * 10.0)  # Penalize high variance
        else:
            stability = 0.5  # Neutral if insufficient data
    else:
        stability = 0.5  # Neutral if no history
    
    # Component 3: Current coverage variance (0.0-1.0)
    coverage_rate = slice_metrics.get("coverage_rate", 0.0)
    abstention_rate = slice_metrics.get("abstention_rate", 0.0)
    
    # Low abstention = more consistent = higher score
    coverage_consistency = max(0.0, 1.0 - abstention_rate)
    
    # Weighted average: homogeneity 30%, stability 40%, consistency 30%
    hss = 0.3 * homogeneity + 0.4 * stability + 0.3 * coverage_consistency
    
    return min(1.0, max(0.0, hss))


def compute_variance_metric(
    slice_metrics: Dict[str, Any],
    historical_metrics: Optional[List[Dict[str, Any]]] = None
) -> float:
    """
    Compute variance metric for a slice (0.0-1.0, lower is better).
    
    Variance captures the instability of a slice's performance over time.
    """
    if not historical_metrics or len(historical_metrics) < 2:
        # Insufficient data - return neutral variance
        return 0.5
    
    # Extract coverage rates
    coverage_rates = [m.get("coverage_rate", 0.0) for m in historical_metrics]
    coverage_rates = [c for c in coverage_rates if c > 0]
    
    if len(coverage_rates) < 2:
        return 0.5
    
    # Compute coefficient of variation
    mean_cov = sum(coverage_rates) / len(coverage_rates)
    if mean_cov == 0:
        return 0.5
    
    variance = sum((c - mean_cov) ** 2 for c in coverage_rates) / len(coverage_rates)
    std_dev = variance ** 0.5
    cv = std_dev / mean_cov
    
    # Normalize CV to 0-1 range (CV > 0.3 is considered high variance)
    variance_metric = min(1.0, cv / 0.3)
    
    return variance_metric


def compute_suitability_score(
    slice_name: str,
    hss: float,
    variance: float,
    slice_metrics: Dict[str, Any]
) -> float:
    """
    Compute overall suitability score for a slice (0.0-1.0).
    
    Suitability indicates whether a slice is fit for production use.
    Low suitability suggests the slice needs attention.
    
    Formula: suitability = (HSS * 0.5) + ((1 - variance) * 0.3) + (coverage * 0.2)
    """
    coverage = slice_metrics.get("coverage_rate", 0.0)
    
    # Weighted combination
    suitability = (hss * 0.5) + ((1.0 - variance) * 0.3) + (coverage * 0.2)
    
    return min(1.0, max(0.0, suitability))


def build_stability_envelope(
    slice_metrics_list: List[Dict[str, Any]],
    historical_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    thresholds: Optional[Dict[str, float]] = None
) -> CurriculumStabilityEnvelope:
    """
    Build a Curriculum Stability Envelope from slice metrics.
    
    Args:
        slice_metrics_list: List of current metrics for each slice
        historical_data: Optional dict mapping slice_name -> list of historical metrics
        thresholds: Optional custom thresholds for assessment
        
    Returns:
        CurriculumStabilityEnvelope with aggregated health metrics
    """
    if thresholds is None:
        thresholds = {
            "hss": DEFAULT_HSS_THRESHOLD,
            "variance": DEFAULT_VARIANCE_THRESHOLD,
            "suitability": DEFAULT_SUITABILITY_THRESHOLD,
        }
    
    historical_data = historical_data or {}
    
    # Compute per-slice health metrics
    slice_health: List[SliceHealthMetrics] = []
    
    for slice_metrics in slice_metrics_list:
        slice_name = slice_metrics.get("slice_name", "unknown")
        history = historical_data.get(slice_name, [])
        
        hss = compute_hss(slice_metrics, history)
        variance = compute_variance_metric(slice_metrics, history)
        suitability = compute_suitability_score(slice_name, hss, variance, slice_metrics)
        
        slice_health.append(SliceHealthMetrics(
            slice_name=slice_name,
            hss=hss,
            variance=variance,
            suitability=suitability,
            coverage_rate=slice_metrics.get("coverage_rate"),
            abstention_rate=slice_metrics.get("abstention_rate"),
        ))
    
    if not slice_health:
        # Empty envelope
        return CurriculumStabilityEnvelope(
            mean_hss=0.0,
            hss_variance=0.0,
            low_hss_fraction=0.0,
            slices_flagged=[],
            suitability_scores={},
            status_light="YELLOW",
        )
    
    # Aggregate metrics
    hss_values = [s.hss for s in slice_health]
    mean_hss = sum(hss_values) / len(hss_values)
    
    # HSS variance
    hss_variance = sum((h - mean_hss) ** 2 for h in hss_values) / len(hss_values)
    hss_std = hss_variance ** 0.5
    
    # Low HSS fraction
    low_hss_count = sum(1 for h in hss_values if h < thresholds["hss"])
    low_hss_fraction = low_hss_count / len(hss_values)
    
    # Identify flagged, stable, unstable slices
    slices_flagged = []
    stable_slices = []
    unstable_slices = []
    hss_variance_spikes = []
    
    for sh in slice_health:
        if sh.suitability < thresholds["suitability"]:
            slices_flagged.append(sh.slice_name)
        
        if sh.hss >= thresholds["hss"] and sh.variance < thresholds["variance"]:
            stable_slices.append(sh.slice_name)
        else:
            unstable_slices.append(sh.slice_name)
        
        if sh.variance > thresholds["variance"]:
            hss_variance_spikes.append(sh.slice_name)
    
    # Determine status light
    if low_hss_fraction > 0.5 or len(slices_flagged) > len(slice_health) // 2:
        status_light = "RED"
    elif low_hss_fraction > 0.25 or len(slices_flagged) > 0:
        status_light = "YELLOW"
    else:
        status_light = "GREEN"
    
    # Build suitability scores dict
    suitability_scores = {sh.slice_name: sh.suitability for sh in slice_health}
    
    return CurriculumStabilityEnvelope(
        mean_hss=mean_hss,
        hss_variance=hss_variance,
        low_hss_fraction=low_hss_fraction,
        slices_flagged=slices_flagged,
        suitability_scores=suitability_scores,
        status_light=status_light,
        stable_slices=stable_slices,
        unstable_slices=unstable_slices,
        hss_variance_spikes=hss_variance_spikes,
    )


def attach_curriculum_stability_to_evidence(
    evidence: Dict[str, Any],
    envelope: CurriculumStabilityEnvelope
) -> Dict[str, Any]:
    """
    Attach curriculum stability envelope to evidence pack (non-mutating).
    
    SHADOW MODE: read-only, non-blocking attachment.
    
    Args:
        evidence: Original evidence dictionary
        envelope: Curriculum stability envelope
        
    Returns:
        New evidence dict with curriculum_stability tile under evidence["governance"]
    """
    # Deep copy to avoid mutation
    import copy
    new_evidence = copy.deepcopy(evidence)
    
    # Ensure governance section exists
    if "governance" not in new_evidence:
        new_evidence["governance"] = {}
    
    # Attach stability tile with limited fields
    new_evidence["governance"]["curriculum_stability"] = {
        "status_light": envelope.status_light,
        "slices_flagged": envelope.slices_flagged,
        "suitability_scores": envelope.suitability_scores,
    }
    
    return new_evidence


def summarize_curriculum_stability_for_council(
    envelope: CurriculumStabilityEnvelope,
    thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Map curriculum stability envelope to Uplift Council advisory.
    
    SHADOW MODE: advisory only, no direct gating.
    
    Args:
        envelope: Curriculum stability envelope
        thresholds: Optional custom thresholds
        
    Returns:
        Council-level advisory dict with status, blocked_slices, marginal_slices
    """
    if thresholds is None:
        thresholds = {
            "suitability": DEFAULT_SUITABILITY_THRESHOLD,
            "critical_suitability": 0.3,  # Below this is critical/blocked
        }
    
    # Classify slices
    blocked_slices = []
    marginal_slices = []
    
    for slice_name, suitability in envelope.suitability_scores.items():
        if suitability < thresholds["critical_suitability"]:
            blocked_slices.append(slice_name)
        elif suitability < thresholds["suitability"]:
            marginal_slices.append(slice_name)
    
    # Determine advisory status
    if blocked_slices:
        status = "BLOCK"
    elif marginal_slices or envelope.status_light == "YELLOW":
        status = "WARN"
    else:
        status = "OK"
    
    return {
        "status": status,
        "blocked_slices": blocked_slices,
        "marginal_slices": marginal_slices,
        "mean_hss": envelope.mean_hss,
        "hss_variance": envelope.hss_variance,
        "status_light": envelope.status_light,
    }


def add_stability_to_first_light(
    first_light_summary: Dict[str, Any],
    envelope: CurriculumStabilityEnvelope
) -> Dict[str, Any]:
    """
    Add curriculum_stability_envelope block to P3 First Light summary.
    
    Args:
        first_light_summary: Existing First Light summary dict
        envelope: Curriculum stability envelope
        
    Returns:
        Updated First Light summary (mutates in-place and returns)
    """
    first_light_summary["curriculum_stability_envelope"] = envelope.to_dict()
    return first_light_summary


def add_stability_to_p4_calibration(
    p4_report: Dict[str, Any],
    envelope: CurriculumStabilityEnvelope
) -> Dict[str, Any]:
    """
    Add curriculum_stability section to P4 calibration report.
    
    SHADOW MODE: observational only, does not block runner.
    
    Args:
        p4_report: Existing P4 calibration report
        envelope: Curriculum stability envelope
        
    Returns:
        Updated P4 report (mutates in-place and returns)
    """
    # Gate decisions: what WOULD be blocked (shadow mode)
    stability_gate_decisions = {}
    for slice_name, suitability in envelope.suitability_scores.items():
        if suitability < DEFAULT_SUITABILITY_THRESHOLD:
            stability_gate_decisions[slice_name] = "BLOCK"
        else:
            stability_gate_decisions[slice_name] = "ALLOW"
    
    p4_report["curriculum_stability"] = {
        "stable_slices": envelope.stable_slices,
        "unstable_slices": envelope.unstable_slices,
        "HSS_variance_spikes": envelope.hss_variance_spikes,
        "stability_gate_decisions": stability_gate_decisions,
    }
    
    return p4_report
