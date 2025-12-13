"""MathLedger curriculum progression package."""

from curriculum.enforcement import (
    # Enums
    DriftSeverity,
    DriftStatus,
    GovernanceSignalType,
    # Data structures
    Violation,
    MonotonicityViolation,
    GateEvolutionViolation,
    ChangedParam,
    CurriculumSnapshot,
    P3VerificationResult,
    DriftTimelineEvent,
    GovernanceSignal,
    # Functions
    verify_curriculum_for_p3,
    # Classes
    DriftTimelineGenerator,
    GovernanceSignalBuilder,
    CurriculumRuntimeGuard,
    # Constants
    GATE_EVOLUTION_RULES,
)

from curriculum.integration import (
    # P3 Stability Hook
    attach_curriculum_governance_to_p3,
    attach_curriculum_timeline_to_p3,
    # Evidence Attachment
    attach_curriculum_to_evidence,
    # Council Classification
    council_classify_curriculum,
    council_classify_curriculum_from_dict,
    summarize_curriculum_for_council,
    # CTRPK
    compute_ctrpk,
    ctrpk_to_status_light,
    compute_ctrpk_trend,
    council_classify_ctrpk,
    build_ctrpk_summary,
    build_ctrpk_compact,
    attach_ctrpk_to_evidence,
)

__all__ = [
    # Enums
    "DriftSeverity",
    "DriftStatus",
    "GovernanceSignalType",
    # Data structures
    "Violation",
    "MonotonicityViolation",
    "GateEvolutionViolation",
    "ChangedParam",
    "CurriculumSnapshot",
    "P3VerificationResult",
    "DriftTimelineEvent",
    "GovernanceSignal",
    # Functions
    "verify_curriculum_for_p3",
    # Classes
    "DriftTimelineGenerator",
    "GovernanceSignalBuilder",
    "CurriculumRuntimeGuard",
    # Constants
    "GATE_EVOLUTION_RULES",
    # P3 Stability Hook
    "attach_curriculum_governance_to_p3",
    "attach_curriculum_timeline_to_p3",
    # Evidence Attachment
    "attach_curriculum_to_evidence",
    # Council Classification
    "council_classify_curriculum",
    "council_classify_curriculum_from_dict",
    "summarize_curriculum_for_council",
    # CTRPK
    "compute_ctrpk",
    "ctrpk_to_status_light",
    "compute_ctrpk_trend",
    "council_classify_ctrpk",
    "build_ctrpk_summary",
    "build_ctrpk_compact",
    "attach_ctrpk_to_evidence",
]

