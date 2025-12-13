"""Backend health module.

Provides health check and monitoring stubs for curriculum drift tiles
and global health canonicalization.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .global_surface import (
    GLOBAL_HEALTH_SURFACE_SCHEMA_VERSION,
    attach_dynamics_tile,
    attach_usla_tile,
    build_global_health_surface,
    clear_usla_producer,
    set_usla_producer,
)

# Policy drift integration
from .policy_drift_tile import (
    attach_policy_drift_tile,
    attach_policy_drift_to_evidence,
    attach_policy_drift_to_p3_stability_report,
    build_policy_drift_summary,
    build_first_light_policy_drift_summary,
    extract_policy_drift_signal_for_first_light,
    policy_drift_vs_nci_for_alignment_view,
    summarize_policy_drift_vs_nci_consistency,
)

# Coherence integration (Phase X)
from .coherence_p3p4_integration import (
    attach_coherence_to_p3_stability_report,
    attach_coherence_to_p4_calibration_report,
    attach_coherence_to_evidence,
    summarize_coherence_for_uplift_council,
)
from .coherence_cal_exp import (
    COHERENCE_SNAPSHOT_SCHEMA_VERSION,
    build_cal_exp_coherence_snapshot,
    persist_coherence_snapshot,
    summarize_coherence_vs_fusion,
    attach_coherence_fusion_crosscheck_to_evidence,
    extract_coherence_fusion_status,
)

# NCI governance integration (Phase X)
from .nci_governance_adapter import (
    NCI_GOVERNANCE_TILE_SCHEMA_VERSION,
    NCI_MODE_DOC_ONLY,
    NCI_MODE_TELEMETRY_CHECKED,
    NCI_MODE_FULLY_BOUND,
    MODE_SLO_THRESHOLDS,
    build_nci_director_panel,
    build_nci_governance_signal,
    build_nci_tile_for_global_health,
    attach_nci_tile_to_global_health,
    check_telemetry_consistency,
    check_slice_consistency,
    build_nci_summary_for_p3,
    attach_nci_summary_to_stability_report,
    attach_nci_to_evidence,
    build_nci_evidence_attachment,
    evaluate_nci_p5,
    contribute_nci_to_ggfl,
    build_ggfl_nci_contribution,
)

# P5 Divergence Interpreter (Phase X)
from .p5_divergence_interpreter import (
    P5DivergenceInterpreter,
    RootCauseHypothesis,
    DiagnosticAction,
    DiagnosticResult,
    AttributionStep,
    interpret_p5_divergence,
    diagnose_from_p4_artifacts,
    build_p5_diagnostic_tile,
    attach_p5_diagnostic_to_evidence,
    p5_diagnostic_for_alignment_view,
)


@dataclass
class HealthStatus:
    """Health check status."""

    status: str = "ok"
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CurriculumDriftTile:
    """Curriculum drift tile data."""

    name: str
    status: str = "ok"
    drift_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


def get_global_health() -> HealthStatus:
    """Get global health status."""
    return HealthStatus(status="ok", message="System healthy")


def canonicalize_health(data: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize health data for consistent output."""
    return {
        "status": data.get("status", "unknown"),
        "timestamp": data.get("timestamp"),
        "components": data.get("components", []),
    }


def build_curriculum_drift_tile(
    name: str,
    drift_events: Optional[List[Dict[str, Any]]] = None,
) -> CurriculumDriftTile:
    """Build curriculum drift tile from events."""
    drift_events = drift_events or []
    drift_score = len(drift_events) * 0.1 if drift_events else 0.0
    return CurriculumDriftTile(
        name=name,
        status="ok" if drift_score < 0.5 else "warn",
        drift_score=drift_score,
        metadata={"event_count": len(drift_events)},
    )


__all__ = [
    # Schema versions
    "COHERENCE_SNAPSHOT_SCHEMA_VERSION",
    "GLOBAL_HEALTH_SURFACE_SCHEMA_VERSION",
    "MODE_SLO_THRESHOLDS",
    "NCI_GOVERNANCE_TILE_SCHEMA_VERSION",
    "NCI_MODE_DOC_ONLY",
    "NCI_MODE_FULLY_BOUND",
    "NCI_MODE_TELEMETRY_CHECKED",
    # Data classes
    "HealthStatus",
    "CurriculumDriftTile",
    # P5 Divergence Interpreter
    "P5DivergenceInterpreter",
    "RootCauseHypothesis",
    "DiagnosticAction",
    "DiagnosticResult",
    "AttributionStep",
    "interpret_p5_divergence",
    "diagnose_from_p4_artifacts",
    "build_p5_diagnostic_tile",
    "attach_p5_diagnostic_to_evidence",
    "p5_diagnostic_for_alignment_view",
    # Coherence
    "attach_coherence_fusion_crosscheck_to_evidence",
    "attach_coherence_to_evidence",
    "extract_coherence_fusion_status",
    "attach_coherence_to_p3_stability_report",
    "attach_coherence_to_p4_calibration_report",
    # Dynamics and USLA
    "attach_dynamics_tile",
    "attach_usla_tile",
    # NCI
    "attach_nci_summary_to_stability_report",
    "attach_nci_tile_to_global_health",
    "attach_nci_to_evidence",
    # Policy drift
    "attach_policy_drift_tile",
    "attach_policy_drift_to_evidence",
    "attach_policy_drift_to_p3_stability_report",
    "build_policy_drift_summary",
    # Build functions
    "build_cal_exp_coherence_snapshot",
    "build_curriculum_drift_tile",
    "build_first_light_policy_drift_summary",
    "build_ggfl_nci_contribution",
    "build_global_health_surface",
    "build_nci_director_panel",
    "build_nci_evidence_attachment",
    "build_nci_governance_signal",
    "build_nci_summary_for_p3",
    "build_nci_tile_for_global_health",
    # Utility functions
    "canonicalize_health",
    "check_slice_consistency",
    "check_telemetry_consistency",
    "clear_usla_producer",
    "contribute_nci_to_ggfl",
    "evaluate_nci_p5",
    "extract_policy_drift_signal_for_first_light",
    "get_global_health",
    "policy_drift_vs_nci_for_alignment_view",
    "persist_coherence_snapshot",
    "set_usla_producer",
    "summarize_policy_drift_vs_nci_consistency",
    "summarize_coherence_for_uplift_council",
    "summarize_coherence_vs_fusion",
]
