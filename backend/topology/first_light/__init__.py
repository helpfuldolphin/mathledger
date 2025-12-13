"""
Phase X P3/P4: First-Light Shadow Experiment

This package implements the First-Light shadow experiment architecture.
See docs/system_law/Phase_X_P3_Spec.md and Phase_X_P4_Spec.md for specifications.

SHADOW MODE CONTRACT:
- No governance control or abort logic is enforced
- All components are observational only
- This code runs OFFLINE only, never in production governance paths

Status:
- P3: IMPLEMENTED (OFFLINE, SHADOW-ONLY)
- P4: DESIGN FREEZE (STUBS ONLY)
"""

from backend.topology.first_light.config import (
    FirstLightConfig,
    FirstLightResult,
)
from backend.topology.first_light.runner import (
    FirstLightShadowRunner,
    CycleObservation,
    SyntheticStateGenerator,
)
from backend.topology.first_light.red_flag_observer import (
    RedFlagType,
    RedFlagSeverity,
    RedFlagObservation,
    RedFlagSummary,
    RedFlagObserver,
)
from backend.topology.first_light.delta_p_computer import (
    DeltaPMetrics,
    DeltaPComputer,
    compute_slope,
)
from backend.topology.first_light.metrics_window import (
    MetricsWindow,
    MetricsAccumulator,
)
from backend.topology.first_light.schemas import (
    CycleLogEntry,
    RedFlagLogEntry,
    MetricsLogEntry,
    SummarySchema,
    CYCLE_LOG_SCHEMA_VERSION,
    RED_FLAG_LOG_SCHEMA_VERSION,
    METRICS_LOG_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
)

# P4 imports (STUBS ONLY - NotImplementedError on all methods)
from backend.topology.first_light.config_p4 import (
    FirstLightConfigP4,
    FirstLightResultP4,
)
from backend.topology.first_light.runner_p4 import (
    FirstLightShadowRunnerP4,
    TwinRunner,
)
from backend.topology.first_light.telemetry_adapter import (
    TelemetryProviderInterface,
    USLAIntegrationAdapter,
    MockTelemetryProvider,
)
from backend.topology.first_light.real_telemetry_adapter import (
    RealTelemetryAdapter,
    RealTelemetryAdapterConfig,
    AdapterMode,
    ValidationResult,
    validate_real_telemetry_window,
    write_trace_jsonl,
    load_trace_jsonl,
)
from backend.topology.first_light.divergence_analyzer import (
    DivergenceAnalyzer,
    DivergenceSummary,
    DivergenceThresholds,
)
from backend.topology.first_light.data_structures_p4 import (
    TelemetrySnapshot,
    RealCycleObservation,
    TwinCycleObservation,
    DivergenceSnapshot,
)
from backend.topology.first_light.schemas_p4 import (
    RealCycleLogEntry,
    TwinCycleLogEntry,
    DivergenceLogEntry,
    P4MetricsLogEntry,
    P4SummarySchema,
    REAL_CYCLE_SCHEMA_VERSION,
    TWIN_CYCLE_SCHEMA_VERSION,
    DIVERGENCE_SCHEMA_VERSION,
    P4_METRICS_SCHEMA_VERSION,
    P4_SUMMARY_SCHEMA_VERSION,
)

# Slice Identity (Phase X pre-execution blocker)
from backend.topology.first_light.slice_identity import (
    InvariantStatus,
    SliceIdentityResult,
    verify_slice_identity_for_p3,
    compute_slice_fingerprint,
    build_identity_console_tile,
    SliceIdentityVerifier,
    # Report/Evidence binding
    attach_slice_identity_to_p3_stability_report,
    attach_slice_identity_to_evidence,
    # P4 drift context
    compute_p4_identity_drift_context,
    P4IdentityDriftContext,
)

# Evidence Pack Builder
from backend.topology.first_light.evidence_pack import (
    build_evidence_pack,
    verify_merkle_root,
    compute_merkle_root,
    detect_status_file,
    detect_p5_divergence_file,
    EvidencePackBuilder,
    EvidencePackResult,
    ArtifactInfo,
    CompletenessCheck,
    GovernanceAdvisory,
    StatusReference,
    P5DivergenceReference,
    EVIDENCE_PACK_VERSION,
    P5_DIVERGENCE_ARTIFACT,
    P5_DIVERGENCE_SCHEMA,
)

# P3 Noise Model Harness
from backend.topology.first_light.noise_harness import (
    P3NoiseConfig,
    P3NoiseModel,
    P3NoiseHarness,
    NoiseDecision,
    NoiseDecisionType,
    NoiseStateSnapshot,
    BaseNoiseParams,
    CorrelatedNoiseParams,
    DegradationParams,
    HeatDeathParams,
    HeavyTailParams,
    NonstationaryParams,
    AdaptiveParams,
    PathologyType,
    PathologySeverity,
    PathologyConfig,
    select_noise_model,
    generate_noise_sample,
    # Noise summary and evidence integration
    build_noise_summary_for_p3,
    attach_noise_to_evidence,
)

# Noise vs Reality Dashboard Generator
from backend.topology.first_light.noise_vs_reality import (
    SCHEMA_VERSION as NOISE_VS_REALITY_SCHEMA_VERSION,
    CoverageVerdict,
    DeltaPScatterPoint,
    RedFlagAnnotation,
    P3SummaryInput,
    P5SummaryInput,
    extract_delta_p_scatter,
    compute_comparison_metrics,
    assess_coverage,
    generate_governance_advisory,
    build_noise_vs_reality_summary,
    build_from_harness_and_divergence,
    validate_noise_vs_reality_summary,
)

__all__ = [
    # Config
    "FirstLightConfig",
    "FirstLightResult",
    # Runner
    "FirstLightShadowRunner",
    "CycleObservation",
    "SyntheticStateGenerator",
    # Red-flag observer
    "RedFlagType",
    "RedFlagSeverity",
    "RedFlagObservation",
    "RedFlagSummary",
    "RedFlagObserver",
    # Delta-p computation
    "DeltaPMetrics",
    "DeltaPComputer",
    "compute_slope",
    # Metrics
    "MetricsWindow",
    "MetricsAccumulator",
    # Schemas
    "CycleLogEntry",
    "RedFlagLogEntry",
    "MetricsLogEntry",
    "SummarySchema",
    "CYCLE_LOG_SCHEMA_VERSION",
    "RED_FLAG_LOG_SCHEMA_VERSION",
    "METRICS_LOG_SCHEMA_VERSION",
    "SUMMARY_SCHEMA_VERSION",
    # P4 Config (STUBS)
    "FirstLightConfigP4",
    "FirstLightResultP4",
    # P4 Runner (STUBS)
    "FirstLightShadowRunnerP4",
    "TwinRunner",
    # P4 Telemetry Adapter (STUBS)
    "TelemetryProviderInterface",
    "USLAIntegrationAdapter",
    "MockTelemetryProvider",
    # P5 Real Telemetry Adapter (POC)
    "RealTelemetryAdapter",
    "RealTelemetryAdapterConfig",
    "AdapterMode",
    "ValidationResult",
    "validate_real_telemetry_window",
    "write_trace_jsonl",
    "load_trace_jsonl",
    # P4 Divergence (STUBS)
    "DivergenceAnalyzer",
    "DivergenceSummary",
    "DivergenceThresholds",
    # P4 Data Structures (STUBS)
    "TelemetrySnapshot",
    "RealCycleObservation",
    "TwinCycleObservation",
    "DivergenceSnapshot",
    # P4 Schemas (STUBS)
    "RealCycleLogEntry",
    "TwinCycleLogEntry",
    "DivergenceLogEntry",
    "P4MetricsLogEntry",
    "P4SummarySchema",
    "REAL_CYCLE_SCHEMA_VERSION",
    "TWIN_CYCLE_SCHEMA_VERSION",
    "DIVERGENCE_SCHEMA_VERSION",
    "P4_METRICS_SCHEMA_VERSION",
    "P4_SUMMARY_SCHEMA_VERSION",
    # Slice Identity (Phase X pre-execution blocker)
    "InvariantStatus",
    "SliceIdentityResult",
    "verify_slice_identity_for_p3",
    "compute_slice_fingerprint",
    "build_identity_console_tile",
    "SliceIdentityVerifier",
    # Slice Identity Report/Evidence binding
    "attach_slice_identity_to_p3_stability_report",
    "attach_slice_identity_to_evidence",
    # Slice Identity P4 drift context
    "compute_p4_identity_drift_context",
    "P4IdentityDriftContext",
    # Evidence Pack Builder
    "build_evidence_pack",
    "verify_merkle_root",
    "compute_merkle_root",
    "detect_status_file",
    "detect_p5_divergence_file",
    "EvidencePackBuilder",
    "EvidencePackResult",
    "ArtifactInfo",
    "CompletenessCheck",
    "GovernanceAdvisory",
    "StatusReference",
    "P5DivergenceReference",
    "EVIDENCE_PACK_VERSION",
    "P5_DIVERGENCE_ARTIFACT",
    "P5_DIVERGENCE_SCHEMA",
    # P3 Noise Model Harness
    "P3NoiseConfig",
    "P3NoiseModel",
    "P3NoiseHarness",
    "NoiseDecision",
    "NoiseDecisionType",
    "NoiseStateSnapshot",
    "BaseNoiseParams",
    "CorrelatedNoiseParams",
    "DegradationParams",
    "HeatDeathParams",
    "HeavyTailParams",
    "NonstationaryParams",
    "AdaptiveParams",
    "PathologyType",
    "PathologySeverity",
    "PathologyConfig",
    "select_noise_model",
    "generate_noise_sample",
    # Noise summary and evidence integration
    "build_noise_summary_for_p3",
    "attach_noise_to_evidence",
    # Noise vs Reality Dashboard Generator
    "NOISE_VS_REALITY_SCHEMA_VERSION",
    "CoverageVerdict",
    "DeltaPScatterPoint",
    "RedFlagAnnotation",
    "P3SummaryInput",
    "P5SummaryInput",
    "extract_delta_p_scatter",
    "compute_comparison_metrics",
    "assess_coverage",
    "generate_governance_advisory",
    "build_noise_vs_reality_summary",
    "build_from_harness_and_divergence",
    "validate_noise_vs_reality_summary",
]
