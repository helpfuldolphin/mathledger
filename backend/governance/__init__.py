"""
Governance and provenance validation module.

This module provides:
- Cryptographic validation of governance chains (attestation lineage)
- Declared roots (block Merkle roots)
- Dual-root integrity (R_t, U_t)
- Last-Mile Governance Checker (CLAUDE K Final Pass Engine)
- Governance audit logging
- Evidence pack collection
- Global Governance Fusion Layer (GGFL) - Phase X SHADOW MODE
"""

from .validator import LawkeeperValidator, GovernanceEntry, DeclaredRoot
from .fusion import (
    build_global_alignment_view,
    EscalationLevel,
    GovernanceAction,
    SIGNAL_PRECEDENCE,
    FUSION_SCHEMA_VERSION,
    Recommendation,
    ConflictDetection,
    FusionResult,
    EscalationState,
    WhatIfAlignmentSignal,
    what_if_for_alignment_view,
)
from .last_mile_checker import (
    GovernanceFinalChecker,
    GovernanceFinalCheckInput,
    GovernanceFinalCheckResult,
    GovernanceFinalCheckConfig,
    GateResult,
    GateEvaluations,
    GovernanceWaiver,
    GovernanceOverride,
    TDAMetrics,
    GateId,
    GateStatus,
    Severity,
    Verdict,
    run_governance_final_check,
)
from .audit_logger import (
    GovernanceAuditLogger,
    AuditLogConfig,
    AuditRecord,
)
from .evidence_pack import (
    GovernanceEvidencePack,
    EvidencePackConfig,
    attach_to_evidence,
    WhatIfStatusSignal,
    detect_what_if_report,
    extract_what_if_status,
    attach_what_if_to_evidence,
    get_what_if_status_from_manifest,
    format_what_if_warning,
    bind_what_if_to_manifest,
)
from .what_if_engine import (
    WhatIfEngine,
    WhatIfCycleInput,
    WhatIfCycleResult,
    WhatIfConfig,
    WhatIfReport,
    GateWhatIfAnalysis,
    NotableEvent,
    CalibrationRecommendation,
    build_what_if_report,
    export_what_if_report,
)

__all__ = [
    # Original exports
    "LawkeeperValidator",
    "GovernanceEntry",
    "DeclaredRoot",
    # Fusion Layer (Phase X SHADOW MODE)
    "build_global_alignment_view",
    "EscalationLevel",
    "GovernanceAction",
    "SIGNAL_PRECEDENCE",
    "FUSION_SCHEMA_VERSION",
    "Recommendation",
    "ConflictDetection",
    "FusionResult",
    "EscalationState",
    # What-If GGFL Adapter
    "WhatIfAlignmentSignal",
    "what_if_for_alignment_view",
    # Last-Mile Checker
    "GovernanceFinalChecker",
    "GovernanceFinalCheckInput",
    "GovernanceFinalCheckResult",
    "GovernanceFinalCheckConfig",
    "GateResult",
    "GateEvaluations",
    "GovernanceWaiver",
    "GovernanceOverride",
    "TDAMetrics",
    "GateId",
    "GateStatus",
    "Severity",
    "Verdict",
    "run_governance_final_check",
    # Audit Logger
    "GovernanceAuditLogger",
    "AuditLogConfig",
    "AuditRecord",
    # Evidence Pack
    "GovernanceEvidencePack",
    "EvidencePackConfig",
    "attach_to_evidence",
    # What-If Auto-Detection & Status (Phase Y)
    "WhatIfStatusSignal",
    "detect_what_if_report",
    "extract_what_if_status",
    "attach_what_if_to_evidence",
    # What-If Manifest Binding
    "get_what_if_status_from_manifest",
    "format_what_if_warning",
    "bind_what_if_to_manifest",
    # What-If Engine (Phase Y Hypothetical Analysis)
    "WhatIfEngine",
    "WhatIfCycleInput",
    "WhatIfCycleResult",
    "WhatIfConfig",
    "WhatIfReport",
    "GateWhatIfAnalysis",
    "NotableEvent",
    "CalibrationRecommendation",
    "build_what_if_report",
    "export_what_if_report",
]
