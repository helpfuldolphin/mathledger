"""Curriculum health monitoring module.

Provides curriculum health checks and Phase V convergence pressure analysis.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


PRESSURE_TENSOR_SCHEMA_VERSION = "1.0.0"


class StatusLight(str, Enum):
    """Status light enum for director tiles."""
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"
    GRAY = "GRAY"


class TransitionLikelihoodBand(str, Enum):
    """Transition likelihood band."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class CurriculumHealthStatus:
    """Curriculum health status."""
    healthy: bool = True
    score: float = 1.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class ConvergencePressure:
    """Phase V convergence pressure metrics."""
    pressure_score: float = 0.0
    converging: bool = True
    cycles_to_convergence: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def check_curriculum_health(
    curriculum: Dict[str, Any],
) -> CurriculumHealthStatus:
    """Check curriculum health."""
    warnings = []
    errors = []

    if not curriculum.get("slices"):
        warnings.append("No slices defined")

    score = 1.0 - (len(warnings) * 0.1 + len(errors) * 0.3)

    return CurriculumHealthStatus(
        healthy=len(errors) == 0,
        score=max(0.0, score),
        warnings=warnings,
        errors=errors,
    )


def compute_convergence_pressure(
    metrics: List[Dict[str, Any]],
) -> ConvergencePressure:
    """Compute Phase V convergence pressure from metrics."""
    if not metrics:
        return ConvergencePressure(pressure_score=0.0, converging=True)

    # Simple pressure calculation
    pressure = len(metrics) * 0.05
    converging = pressure < 0.8

    return ConvergencePressure(
        pressure_score=min(1.0, pressure),
        converging=converging,
        cycles_to_convergence=10 if converging else None,
        metadata={"metric_count": len(metrics)},
    )


def validate_curriculum_phase_v(
    curriculum: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate curriculum for Phase V requirements."""
    health = check_curriculum_health(curriculum)

    return {
        "valid": health.healthy,
        "score": health.score,
        "warnings": health.warnings,
        "errors": health.errors,
        "phase": "V",
    }


def build_convergence_pressure_tensor(
    convergence_map: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build convergence pressure tensor from convergence map."""
    config = config or {}

    slices_converging = convergence_map.get("slices_converging", [])
    slices_diverging = convergence_map.get("slices_diverging", [])
    correlations = convergence_map.get("cross_signal_correlations", {})

    # Compute pressure components
    divergence_pressure = len(slices_diverging) * 0.2
    convergence_support = len(slices_converging) * 0.1
    correlation_pressure = sum(correlations.values()) / max(1, len(correlations))

    overall_pressure = min(1.0, divergence_pressure - convergence_support + correlation_pressure)

    return {
        "schema_version": PRESSURE_TENSOR_SCHEMA_VERSION,
        "overall_pressure": max(0.0, overall_pressure),
        "divergence_pressure": divergence_pressure,
        "convergence_support": convergence_support,
        "correlation_pressure": correlation_pressure,
        "slices_converging": slices_converging,
        "slices_diverging": slices_diverging,
    }


def build_phase_transition_early_warning_radar(
    convergence_map: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Build phase transition early warning radar."""
    thresholds = thresholds or {"pressure": 0.5, "velocity": 0.3}

    tensor = build_convergence_pressure_tensor(convergence_map)
    pressure = tensor["overall_pressure"]

    # Determine likelihood band
    if pressure < 0.2:
        band = TransitionLikelihoodBand.LOW
    elif pressure < 0.5:
        band = TransitionLikelihoodBand.MEDIUM
    elif pressure < 0.8:
        band = TransitionLikelihoodBand.HIGH
    else:
        band = TransitionLikelihoodBand.CRITICAL

    warnings = []
    if pressure > thresholds.get("pressure", 0.5):
        warnings.append("High pressure detected")

    return {
        "pressure_score": pressure,
        "likelihood_band": band.value,
        "warnings": warnings,
        "thresholds": thresholds,
        "status": "ok" if band in (TransitionLikelihoodBand.LOW, TransitionLikelihoodBand.MEDIUM) else "warn",
        "root_drivers": tensor.get("slices_diverging", []),
    }


def build_convergence_director_tile(
    convergence_map: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build convergence director tile for dashboard."""
    tensor = build_convergence_pressure_tensor(convergence_map, config)
    pressure = tensor["overall_pressure"]

    # Determine status light
    if pressure < 0.3:
        status_light = StatusLight.GREEN
    elif pressure < 0.6:
        status_light = StatusLight.YELLOW
    else:
        status_light = StatusLight.RED

    return {
        "pressure_score": pressure,
        "converging": pressure < 0.5,
        "status_light": status_light.value,
        "status": "ok" if status_light == StatusLight.GREEN else "warn",
    }


__all__ = [
    "PRESSURE_TENSOR_SCHEMA_VERSION",
    "StatusLight",
    "TransitionLikelihoodBand",
    "CurriculumHealthStatus",
    "ConvergencePressure",
    "check_curriculum_health",
    "compute_convergence_pressure",
    "validate_curriculum_phase_v",
    "build_convergence_pressure_tensor",
    "build_phase_transition_early_warning_radar",
    "build_convergence_director_tile",
]
