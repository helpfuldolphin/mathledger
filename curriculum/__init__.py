"""MathLedger curriculum progression package."""

# Phase II uplift curriculum loader and drift detection
from curriculum.phase2_loader import (
    CurriculumLoaderV2,
    SuccessMetricSpec,
    UpliftSlice,
    CurriculumFingerprint,
    compute_curriculum_diff,
)

from curriculum.drift_radar import (
    build_curriculum_drift_history,
    classify_curriculum_drift_event,
    evaluate_curriculum_for_promotion,
    summarize_curriculum_for_global_health,
)

__all__ = [
    "CurriculumLoaderV2",
    "SuccessMetricSpec",
    "UpliftSlice",
    "CurriculumFingerprint",
    "compute_curriculum_diff",
    "build_curriculum_drift_history",
    "classify_curriculum_drift_event",
    "evaluate_curriculum_for_promotion",
    "summarize_curriculum_for_global_health",
]

