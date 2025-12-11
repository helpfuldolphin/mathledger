"""MathLedger curriculum progression package."""

from .stability import (
    CurriculumStabilityEnvelope,
    SliceHealthMetrics,
    build_stability_envelope,
    compute_hss,
    compute_variance_metric,
    compute_suitability_score,
    attach_curriculum_stability_to_evidence,
    summarize_curriculum_stability_for_council,
    add_stability_to_first_light,
    add_stability_to_p4_calibration,
)

from .integration import (
    add_stability_to_rfl_results,
    add_stability_to_u2_results,
    create_p4_calibration_report_with_stability,
    create_evidence_pack_with_stability,
)

__all__ = [
    "CurriculumStabilityEnvelope",
    "SliceHealthMetrics",
    "build_stability_envelope",
    "compute_hss",
    "compute_variance_metric",
    "compute_suitability_score",
    "attach_curriculum_stability_to_evidence",
    "summarize_curriculum_stability_for_council",
    "add_stability_to_first_light",
    "add_stability_to_p4_calibration",
    "add_stability_to_rfl_results",
    "add_stability_to_u2_results",
    "create_p4_calibration_report_with_stability",
    "create_evidence_pack_with_stability",
]

