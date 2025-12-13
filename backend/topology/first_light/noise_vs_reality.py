"""
P3/P5 Noise vs Reality Dashboard Generator

Implements the noise_vs_reality_summary generator per STRATCOM execution order:
- p3_summary: Extracted from P3 noise harness data
- p5_summary: Extracted from P5 real telemetry divergence data
- comparison_metrics: Coverage ratio, exceedance analysis, correlation
- coverage_assessment: ADEQUATE/MARGINAL/INSUFFICIENT verdict

SHADOW MODE CONTRACT:
- All operations are observational only
- No governance modification
- Generator produces comparison data, never modifies source

Schema Version: noise-vs-reality/1.0.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# Constants and Enums
# =============================================================================

SCHEMA_VERSION = "noise-vs-reality/1.0.0"


class CoverageVerdict(Enum):
    """Coverage assessment verdict."""
    ADEQUATE = "ADEQUATE"
    MARGINAL = "MARGINAL"
    INSUFFICIENT = "INSUFFICIENT"


class AdvisorySeverity(Enum):
    """Governance advisory severity (SHADOW mode only)."""
    INFO = "INFO"
    WARN = "WARN"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DeltaPScatterPoint:
    """Single point in Δp scatter plot."""
    cycle: int
    twin_delta_p: float
    real_delta_p: float
    divergence_magnitude: float
    is_red_flag: bool = False
    red_flag_type: Optional[str] = None


@dataclass
class RedFlagAnnotation:
    """Red-flag annotation for scatter plot."""
    cycle: int
    red_flag_type: str
    severity: str
    description: str


@dataclass
class P3SummaryInput:
    """
    Input data for P3 summary extraction.

    Can be populated from P3NoiseHarness.get_stability_report() and
    build_noise_summary_for_p3() output.
    """
    total_cycles: int
    noise_event_rate: float
    regime_proportions: Dict[str, float]
    delta_p_contribution: Dict[str, Any]
    pathology_events: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_noise_summary(cls, noise_summary: Dict[str, Any]) -> "P3SummaryInput":
        """Create from build_noise_summary_for_p3() output."""
        total_cycles = noise_summary.get("total_cycles", 0)

        # Extract noise event rate from RSI aggregate
        rsi_agg = noise_summary.get("rsi_aggregate", {})
        noise_event_rate = rsi_agg.get("noise_event_rate", 0.0)

        # Extract regime proportions
        regime_proportions = noise_summary.get("regime_proportions", {})

        # Extract delta_p contribution
        delta_p_agg = noise_summary.get("delta_p_aggregate", {})
        delta_p_contribution = {
            "total": delta_p_agg.get("total_contribution", 0.0),
            "by_type": delta_p_agg.get("by_noise_type", {}),
        }

        return cls(
            total_cycles=total_cycles,
            noise_event_rate=noise_event_rate,
            regime_proportions=regime_proportions,
            delta_p_contribution=delta_p_contribution,
            pathology_events=[],
        )


@dataclass
class P5SummaryInput:
    """
    Input data for P5 summary extraction.

    Populated from real telemetry divergence data.
    """
    total_cycles: int
    divergence_time_series: List[DeltaPScatterPoint]
    red_flags: List[RedFlagAnnotation]
    telemetry_source: Dict[str, str]

    @classmethod
    def from_divergence_data(
        cls,
        divergence_series: List[Dict[str, Any]],
        red_flags: Optional[List[Dict[str, Any]]] = None,
        provider: str = "usla_adapter",
        start_ts: Optional[str] = None,
        end_ts: Optional[str] = None,
    ) -> "P5SummaryInput":
        """Create from raw divergence data."""
        now = datetime.now(timezone.utc).isoformat()

        # Convert divergence series
        scatter_points = []
        for entry in divergence_series:
            point = DeltaPScatterPoint(
                cycle=entry.get("cycle", 0),
                twin_delta_p=entry.get("twin_delta_p", 0.0),
                real_delta_p=entry.get("real_delta_p", 0.0),
                divergence_magnitude=entry.get("divergence_magnitude", 0.0),
                is_red_flag=entry.get("is_red_flag", False),
                red_flag_type=entry.get("red_flag_type"),
            )
            scatter_points.append(point)

        # Convert red flags
        annotations = []
        for rf in (red_flags or []):
            ann = RedFlagAnnotation(
                cycle=rf.get("cycle", 0),
                red_flag_type=rf.get("type", "UNKNOWN"),
                severity=rf.get("severity", "WARN"),
                description=rf.get("description", ""),
            )
            annotations.append(ann)

        return cls(
            total_cycles=len(scatter_points),
            divergence_time_series=scatter_points,
            red_flags=annotations,
            telemetry_source={
                "provider": provider,
                "start_timestamp": start_ts or now,
                "end_timestamp": end_ts or now,
            },
        )


# =============================================================================
# Δp Scatter Extractor
# =============================================================================

def extract_delta_p_scatter(
    p5_input: P5SummaryInput,
    divergence_threshold: float = 0.01,
) -> Dict[str, Any]:
    """
    Extract Δp scatter data with red-flag annotations.

    Implements: Δp scatter extractor per STRATCOM task.

    Args:
        p5_input: P5 summary input with divergence time series
        divergence_threshold: Threshold for "significant" divergence

    Returns:
        Dict with scatter points and red-flag annotations
    """
    scatter_points = []
    red_flag_cycles = []

    for point in p5_input.divergence_time_series:
        scatter_points.append({
            "cycle": point.cycle,
            "twin_delta_p": round(point.twin_delta_p, 6),
            "real_delta_p": round(point.real_delta_p, 6),
            "divergence_magnitude": round(point.divergence_magnitude, 6),
            "is_red_flag": point.is_red_flag,
        })

        if point.is_red_flag:
            red_flag_cycles.append(point.cycle)

    # Build red-flag annotations
    annotations = []
    for rf in p5_input.red_flags:
        annotations.append({
            "cycle": rf.cycle,
            "type": rf.red_flag_type,
            "severity": rf.severity,
            "description": rf.description,
        })

    # Compute statistics
    if scatter_points:
        divergences = [p["divergence_magnitude"] for p in scatter_points]
        significant_count = sum(1 for d in divergences if abs(d) > divergence_threshold)

        stats = {
            "mean": round(sum(divergences) / len(divergences), 6),
            "std": round(_std(divergences), 6),
            "max": round(max(abs(d) for d in divergences), 6),
            "p95": round(_percentile(divergences, 95), 6),
            "significant_divergence_count": significant_count,
            "divergence_rate": round(significant_count / len(scatter_points), 4),
        }
    else:
        stats = {
            "mean": 0.0,
            "std": 0.0,
            "max": 0.0,
            "p95": 0.0,
            "significant_divergence_count": 0,
            "divergence_rate": 0.0,
        }

    return {
        "scatter_points": scatter_points,
        "red_flag_annotations": annotations,
        "red_flag_cycles": red_flag_cycles,
        "statistics": stats,
    }


def _std(values: List[float]) -> float:
    """Compute standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def _percentile(values: List[float], pct: float) -> float:
    """Compute percentile."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * pct / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


# =============================================================================
# Comparison Metrics Calculator
# =============================================================================

def compute_comparison_metrics(
    p3_summary: Dict[str, Any],
    p5_summary: Dict[str, Any],
    p3_noise_bounds: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute comparison metrics between P3 and P5.

    Args:
        p3_summary: P3 summary dict with noise_event_rate
        p5_summary: P5 summary dict with divergence_rate
        p3_noise_bounds: Optional max noise impact for exceedance check

    Returns:
        comparison_metrics dict per schema
    """
    p3_noise_rate = p3_summary.get("noise_event_rate", 0.0)
    p5_divergence_rate = p5_summary.get("divergence_rate", 0.0)

    # Coverage ratio (handle div-by-zero)
    if p5_divergence_rate > 0:
        coverage_ratio = p3_noise_rate / p5_divergence_rate
    else:
        coverage_ratio = float('inf') if p3_noise_rate > 0 else 1.0

    # Noise vs divergence rate comparison
    noise_vs_divergence = {
        "p3_noise_rate": round(p3_noise_rate, 4),
        "p5_divergence_rate": round(p5_divergence_rate, 4),
        "difference": round(p3_noise_rate - p5_divergence_rate, 4),
    }

    # Delta-p correlation (requires time series alignment)
    # For now, use simplified correlation from aggregate stats
    p3_delta_p_total = p3_summary.get("delta_p_contribution", {}).get("total", 0.0)
    p5_divergence_stats = p5_summary.get("divergence_stats", {})
    p5_mean_divergence = p5_divergence_stats.get("mean", 0.0)

    # Simplified correlation interpretation
    if abs(p3_delta_p_total) < 0.001 and abs(p5_mean_divergence) < 0.001:
        pearson_r = 0.0
        interpretation = "Both P3 noise impact and P5 divergence are minimal"
    elif p3_delta_p_total * p5_mean_divergence > 0:
        pearson_r = 0.3  # Weak positive (same direction)
        interpretation = "Weak positive correlation; P3 noise partially explains P5 variance"
    elif p3_delta_p_total * p5_mean_divergence < 0:
        pearson_r = -0.2  # Weak negative (opposite direction)
        interpretation = "Weak negative correlation; P3 noise and P5 divergence have opposite effects"
    else:
        pearson_r = 0.1
        interpretation = "Minimal correlation between P3 noise and P5 divergence"

    delta_p_correlation = {
        "pearson_r": round(pearson_r, 3),
        "interpretation": interpretation,
    }

    # Exceedance analysis
    exceedance_cycles = []
    exceedance_rate = 0.0

    # Check if P5 divergence exceeds P3 synthetic bounds
    p3_max_noise_impact = p3_noise_bounds or abs(p3_delta_p_total) / max(1, p3_summary.get("total_cycles", 1))
    p5_divergence_series = p5_summary.get("divergence_time_series", [])

    if p5_divergence_series and p3_max_noise_impact > 0:
        for point in p5_divergence_series:
            div_mag = point.get("divergence_magnitude", 0.0)
            if abs(div_mag) > p3_max_noise_impact * 2:  # 2x threshold for exceedance
                exceedance_cycles.append(point.get("cycle", 0))
        exceedance_rate = len(exceedance_cycles) / len(p5_divergence_series)

    return {
        "coverage_ratio": round(coverage_ratio, 4) if coverage_ratio != float('inf') else 999.99,
        "noise_vs_divergence_rate": noise_vs_divergence,
        "delta_p_correlation": delta_p_correlation,
        "exceedance_cycles": exceedance_cycles,
        "exceedance_rate": round(exceedance_rate, 4),
    }


# =============================================================================
# Coverage Assessment
# =============================================================================

def assess_coverage(
    comparison_metrics: Dict[str, Any],
    p3_summary: Dict[str, Any],
    p5_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Assess whether P3 adequately covers P5 behavior.

    Verdict logic:
    - ADEQUATE: coverage_ratio >= 1.0 and exceedance_rate < 0.05
    - MARGINAL: coverage_ratio >= 0.8 or exceedance_rate < 0.10
    - INSUFFICIENT: otherwise

    Args:
        comparison_metrics: Output from compute_comparison_metrics
        p3_summary: P3 summary dict
        p5_summary: P5 summary dict

    Returns:
        coverage_assessment dict per schema
    """
    coverage_ratio = comparison_metrics.get("coverage_ratio", 0.0)
    exceedance_rate = comparison_metrics.get("exceedance_rate", 0.0)

    # Determine verdict
    if coverage_ratio >= 1.0 and exceedance_rate < 0.05:
        verdict = CoverageVerdict.ADEQUATE
        confidence = min(0.95, 0.7 + (coverage_ratio - 1.0) * 0.1 + (0.05 - exceedance_rate) * 2)
    elif coverage_ratio >= 0.8 or exceedance_rate < 0.10:
        verdict = CoverageVerdict.MARGINAL
        confidence = 0.5 + min(0.3, coverage_ratio * 0.2)
    else:
        verdict = CoverageVerdict.INSUFFICIENT
        confidence = max(0.2, 0.5 - (1.0 - coverage_ratio) * 0.3)

    # Generate reasoning
    reasoning_parts = []

    p3_noise_rate = comparison_metrics.get("noise_vs_divergence_rate", {}).get("p3_noise_rate", 0)
    p5_div_rate = comparison_metrics.get("noise_vs_divergence_rate", {}).get("p5_divergence_rate", 0)

    if coverage_ratio >= 1.0:
        reasoning_parts.append(
            f"P3 synthetic noise rate ({p3_noise_rate*100:.1f}%) exceeds "
            f"P5 observed divergence rate ({p5_div_rate*100:.1f}%) "
            f"with coverage ratio {coverage_ratio:.2f}."
        )
    else:
        reasoning_parts.append(
            f"P3 synthetic noise rate ({p3_noise_rate*100:.1f}%) is below "
            f"P5 observed divergence rate ({p5_div_rate*100:.1f}%) "
            f"with coverage ratio {coverage_ratio:.2f}."
        )

    exceedance_count = len(comparison_metrics.get("exceedance_cycles", []))
    p5_total = p5_summary.get("total_cycles", 0)
    if exceedance_count > 0:
        reasoning_parts.append(
            f"{exceedance_count} cycle(s) ({exceedance_rate*100:.1f}%) showed "
            f"P5 divergence exceeding P3 bounds."
        )
    else:
        reasoning_parts.append("No cycles showed P5 divergence exceeding P3 bounds.")

    # Identify gaps
    gaps = []
    p5_red_flags = p5_summary.get("red_flags", {})
    p5_red_flag_types = p5_red_flags.get("types", {})
    p3_pathologies = p3_summary.get("pathology_events", [])
    p3_pathology_types = set(p.get("type", "") for p in p3_pathologies)

    # Check if P5 red-flag types are covered by P3 pathologies
    for rf_type, count in p5_red_flag_types.items():
        if rf_type not in p3_pathology_types:
            gaps.append({
                "description": f"Red-flag type '{rf_type}' observed in P5 but not modeled in P3",
                "p5_frequency": count / max(1, p5_total),
                "suggested_p3_pathology": _suggest_pathology(rf_type),
            })

    # Generate recommendations
    recommendations = []

    if verdict == CoverageVerdict.INSUFFICIENT:
        recommendations.append("Increase P3 noise rates or add pathology profiles to better bracket P5 behavior")

    if gaps:
        recommendations.append(f"Consider adding P3 pathologies for: {', '.join(g['suggested_p3_pathology'] for g in gaps)}")

    if exceedance_rate > 0.05:
        recommendations.append("Review exceedance cycles for patterns not captured by P3 noise model")

    if not recommendations:
        recommendations.append("Continue monitoring; current P3 model adequately covers P5 behavior")

    return {
        "verdict": verdict.value,
        "confidence": round(confidence, 2),
        "reasoning": " ".join(reasoning_parts),
        "gaps": gaps,
        "recommendations": recommendations,
    }


def _suggest_pathology(red_flag_type: str) -> str:
    """Suggest P3 pathology based on red-flag type."""
    suggestions = {
        "DELTA_P_SPIKE": "spike",
        "DELTA_P_DROP": "spike",
        "RSI_DROP": "drift",
        "RSI_SPIKE": "spike",
        "TIMEOUT_CLUSTER": "cluster_burst",
        "RESOURCE_EXHAUSTION": "heat_death_cascade",
        "OSCILLATION": "oscillation",
    }
    return suggestions.get(red_flag_type, "spike")


# =============================================================================
# Governance Advisory Generator
# =============================================================================

def generate_governance_advisory(
    coverage_assessment: Dict[str, Any],
    comparison_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate governance advisory based on coverage assessment.

    SHADOW MODE: Severity is always INFO or WARN, never CRITICAL/BLOCK.

    SINGLE WARNING CAP: At most one warning line with top driving factor.

    Args:
        coverage_assessment: Output from assess_coverage
        comparison_metrics: Optional comparison metrics for top factor details

    Returns:
        governance_advisory dict per schema with:
        - severity: INFO or WARN
        - message: Single line with top factor if MARGINAL/INSUFFICIENT
        - top_factor: The primary driver (coverage_ratio or exceedance_rate)
        - top_factor_value: Numeric value of the top factor
        - action_required: Boolean
    """
    verdict = coverage_assessment.get("verdict", "ADEQUATE")
    confidence = coverage_assessment.get("confidence", 1.0)

    # Extract top factor from comparison_metrics or coverage_assessment
    coverage_ratio = None
    exceedance_rate = None

    if comparison_metrics:
        coverage_ratio = comparison_metrics.get("coverage_ratio")
        exceedance_rate = comparison_metrics.get("exceedance_rate", 0.0)
    else:
        # Try to get from assessment reasoning
        coverage_ratio = coverage_assessment.get("confidence", 1.0)
        exceedance_rate = 0.0

    # Determine top factor driving the verdict
    top_factor = None
    top_factor_value = None

    if verdict == "ADEQUATE":
        return {
            "severity": AdvisorySeverity.INFO.value,
            "message": "P3 synthetic stress adequately brackets observed P5 divergence.",
            "top_factor": None,
            "top_factor_value": None,
            "action_required": False,
        }

    # For MARGINAL/INSUFFICIENT, identify the top driving factor
    # Priority: lower coverage_ratio or higher exceedance_rate
    if coverage_ratio is not None and exceedance_rate is not None:
        # Coverage ratio < 1.0 means P3 doesn't fully cover P5
        # Exceedance rate > 0 means some cycles exceed bounds
        coverage_deficit = max(0, 1.0 - coverage_ratio) if coverage_ratio else 0
        exceedance_severity = exceedance_rate if exceedance_rate else 0

        if coverage_deficit >= exceedance_severity:
            top_factor = "coverage_ratio"
            top_factor_value = coverage_ratio
        else:
            top_factor = "exceedance_rate"
            top_factor_value = exceedance_rate
    elif coverage_ratio is not None:
        top_factor = "coverage_ratio"
        top_factor_value = coverage_ratio
    elif exceedance_rate is not None:
        top_factor = "exceedance_rate"
        top_factor_value = exceedance_rate

    # Build SINGLE warning message with top factor
    if verdict == "MARGINAL":
        if top_factor == "coverage_ratio" and top_factor_value is not None:
            message = f"P3 coverage marginal: coverage_ratio={top_factor_value:.2f}"
        elif top_factor == "exceedance_rate" and top_factor_value is not None:
            message = f"P3 coverage marginal: exceedance_rate={top_factor_value*100:.1f}%"
        else:
            message = "P3 coverage is marginal; some P5 behavior may not be fully modeled."
        return {
            "severity": AdvisorySeverity.WARN.value,
            "message": message,
            "top_factor": top_factor,
            "top_factor_value": round(top_factor_value, 4) if top_factor_value else None,
            "action_required": False,
        }

    # INSUFFICIENT
    if top_factor == "coverage_ratio" and top_factor_value is not None:
        message = f"P3 coverage insufficient: coverage_ratio={top_factor_value:.2f}"
    elif top_factor == "exceedance_rate" and top_factor_value is not None:
        message = f"P3 coverage insufficient: exceedance_rate={top_factor_value*100:.1f}%"
    else:
        message = "P3 coverage is insufficient; P5 divergence exceeds synthetic stress bounds."
    return {
        "severity": AdvisorySeverity.WARN.value,
        "message": message,
        "top_factor": top_factor,
        "top_factor_value": round(top_factor_value, 4) if top_factor_value else None,
        "action_required": True,
    }


# =============================================================================
# Main Generator
# =============================================================================

def build_noise_vs_reality_summary(
    p3_input: P3SummaryInput,
    p5_input: P5SummaryInput,
    experiment_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build complete noise_vs_reality_summary.

    Implements: Generator producing p3_summary, p5_summary, comparison_metrics,
    coverage_assessment per STRATCOM execution order.

    Args:
        p3_input: P3 summary input data
        p5_input: P5 summary input data
        experiment_id: Optional experiment identifier

    Returns:
        Complete noise_vs_reality_summary matching schema noise-vs-reality/1.0.0
    """
    now = datetime.now(timezone.utc).isoformat()
    exp_id = experiment_id or f"nvr-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    # Build P3 summary
    p3_summary = {
        "total_cycles": p3_input.total_cycles,
        "noise_event_rate": round(p3_input.noise_event_rate, 4),
        "regime_proportions": p3_input.regime_proportions,
        "delta_p_contribution": p3_input.delta_p_contribution,
        "pathology_events": p3_input.pathology_events,
    }

    # Extract Δp scatter and build P5 summary
    scatter_result = extract_delta_p_scatter(p5_input)

    # Build red-flag summary
    red_flag_types: Dict[str, int] = {}
    for ann in scatter_result["red_flag_annotations"]:
        rf_type = ann["type"]
        red_flag_types[rf_type] = red_flag_types.get(rf_type, 0) + 1

    p5_summary = {
        "total_cycles": p5_input.total_cycles,
        "divergence_rate": scatter_result["statistics"]["divergence_rate"],
        "divergence_stats": {
            "mean": scatter_result["statistics"]["mean"],
            "std": scatter_result["statistics"]["std"],
            "max": scatter_result["statistics"]["max"],
            "p95": scatter_result["statistics"]["p95"],
        },
        "red_flags": {
            "count": len(scatter_result["red_flag_cycles"]),
            "types": red_flag_types,
            "cycles": scatter_result["red_flag_cycles"],
        },
        "telemetry_source": p5_input.telemetry_source,
        "divergence_time_series": scatter_result["scatter_points"],
    }

    # Compute comparison metrics
    comparison_metrics = compute_comparison_metrics(p3_summary, p5_summary)

    # Assess coverage
    coverage_assessment = assess_coverage(comparison_metrics, p3_summary, p5_summary)

    # Generate governance advisory (with comparison_metrics for top factor)
    governance_advisory = generate_governance_advisory(coverage_assessment, comparison_metrics)

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": exp_id,
        "generated_at": now,
        "mode": "SHADOW",
        "p3_summary": p3_summary,
        "p5_summary": p5_summary,
        "comparison_metrics": comparison_metrics,
        "coverage_assessment": coverage_assessment,
        "governance_advisory": governance_advisory,
    }


# =============================================================================
# Convenience Functions
# =============================================================================

def build_from_harness_and_divergence(
    noise_summary: Dict[str, Any],
    divergence_series: List[Dict[str, Any]],
    red_flags: Optional[List[Dict[str, Any]]] = None,
    experiment_id: Optional[str] = None,
    telemetry_provider: str = "usla_adapter",
) -> Dict[str, Any]:
    """
    Build noise_vs_reality_summary from noise harness summary and divergence data.

    Convenience wrapper for integration with existing P3/P5 pipelines.

    Args:
        noise_summary: Output from build_noise_summary_for_p3()
        divergence_series: List of divergence points (cycle, twin_delta_p, real_delta_p, ...)
        red_flags: Optional list of red-flag annotations
        experiment_id: Optional experiment identifier
        telemetry_provider: Name of telemetry provider

    Returns:
        Complete noise_vs_reality_summary
    """
    p3_input = P3SummaryInput.from_noise_summary(noise_summary)
    p5_input = P5SummaryInput.from_divergence_data(
        divergence_series=divergence_series,
        red_flags=red_flags,
        provider=telemetry_provider,
    )

    return build_noise_vs_reality_summary(p3_input, p5_input, experiment_id)


def validate_noise_vs_reality_summary(summary: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate noise_vs_reality_summary against schema requirements.

    Args:
        summary: Summary dict to validate

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    # Required top-level fields
    required_fields = [
        "schema_version",
        "experiment_id",
        "generated_at",
        "p3_summary",
        "p5_summary",
        "comparison_metrics",
        "coverage_assessment",
    ]

    for field in required_fields:
        if field not in summary:
            errors.append(f"Missing required field: {field}")

    # Check schema version
    if summary.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"Invalid schema_version: expected {SCHEMA_VERSION}")

    # Check mode is SHADOW
    if summary.get("mode") != "SHADOW":
        errors.append("mode must be 'SHADOW'")

    # Validate P3 summary
    p3 = summary.get("p3_summary", {})
    p3_required = ["total_cycles", "noise_event_rate", "regime_proportions", "delta_p_contribution"]
    for field in p3_required:
        if field not in p3:
            errors.append(f"Missing p3_summary field: {field}")

    # Validate P5 summary
    p5 = summary.get("p5_summary", {})
    p5_required = ["total_cycles", "divergence_rate", "divergence_stats", "red_flags"]
    for field in p5_required:
        if field not in p5:
            errors.append(f"Missing p5_summary field: {field}")

    # Validate comparison metrics
    cm = summary.get("comparison_metrics", {})
    cm_required = ["coverage_ratio", "noise_vs_divergence_rate", "delta_p_correlation"]
    for field in cm_required:
        if field not in cm:
            errors.append(f"Missing comparison_metrics field: {field}")

    # Validate coverage assessment
    ca = summary.get("coverage_assessment", {})
    ca_required = ["verdict", "confidence", "gaps"]
    for field in ca_required:
        if field not in ca:
            errors.append(f"Missing coverage_assessment field: {field}")

    # Validate verdict value
    if ca.get("verdict") not in ["ADEQUATE", "MARGINAL", "INSUFFICIENT"]:
        errors.append(f"Invalid verdict: {ca.get('verdict')}")

    # Validate ranges
    if p3:
        rate = p3.get("noise_event_rate", 0)
        if not (0.0 <= rate <= 1.0):
            errors.append(f"noise_event_rate out of range [0,1]: {rate}")

    if ca:
        conf = ca.get("confidence", 0)
        if not (0.0 <= conf <= 1.0):
            errors.append(f"confidence out of range [0,1]: {conf}")

    return len(errors) == 0, errors
