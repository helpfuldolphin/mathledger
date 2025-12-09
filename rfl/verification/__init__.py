"""
RFL Verification Subsystem

This package contains verification-related modules for the Reflexive Formal Learning
(RFL) system, including abstention taxonomy and verification outcome classification.

Modules:
    abstention_taxonomy: Canonical abstention type enumeration and classification utilities
    failure_classifier: Unified failure-state classification for execution failures
    abstention_record: Unified AbstentionRecord bridging verification and experiment layers
    abstention_semantics: Semantic tree, categories, and JSON schema validation
"""

from .abstention_taxonomy import (
    AbstentionType,
    classify_verification_method,
    classify_breakdown_key,
    serialize_abstention,
    deserialize_abstention,
    ABSTENTION_METHOD_STRINGS,
    is_abstention_method,
)

from .failure_classifier import (
    FailureState,
    classify_exception,
    classify_and_log,
    classify_from_result,
    classify_from_status,
    normalize_legacy_key,
    failure_to_abstention,
    classify_breakdown_key_unified,
)

from .abstention_record import (
    AbstentionRecord,
    CANONICAL_ABSTENTION_ORDER,
    CANONICAL_ABSTENTION_SET,
    GLOBAL_VALIDATION_ENABLED,
    create_canonical_histogram,
    normalize_histogram,
    merge_histograms,
    histogram_to_records,
)

from .abstention_semantics import (
    # Semantic tree
    ABSTENTION_TREE,
    SemanticCategory,
    categorize,
    get_category,
    get_types_for_category,
    get_all_categories,
    is_timeout_related,
    is_crash_related,
    is_resource_related,
    is_oracle_related,
    # Schema
    ABSTENTION_RECORD_SCHEMA,
    get_schema,
    get_schema_path,
    # Validation
    AbstentionValidationError,
    validate_abstention_data,
    validate_abstention_record,
    validate_abstention_json,
    # Aggregation
    aggregate_by_category,
    aggregate_histogram_by_category,
    # Analytics (Phase II v1.1)
    summarize_abstentions,
    detect_abstention_red_flags,
    # Phase III: Red-Flag Feed & Global Health
    HEALTH_SNAPSHOT_SCHEMA_VERSION,
    build_abstention_health_snapshot,
    build_abstention_radar,
    summarize_abstentions_for_uplift,
    summarize_abstentions_for_global_health,
    # Phase IV: Epistemic Risk Decomposition & Cross-Signal Integration
    EPISTEMIC_PROFILE_SCHEMA_VERSION,
    build_epistemic_abstention_profile,
    compose_abstention_with_budget_and_perf,
    build_abstention_director_panel,
    build_abstention_storyline,
    evaluate_abstention_for_uplift,
    # Phase V: Double-Helix Drift Radar & Global Governance Linkage
    DRIFT_TIMELINE_SCHEMA_VERSION,
    build_epistemic_drift_timeline,
    summarize_abstention_for_global_console,
    compose_abstention_with_uplift_decision,
    # Governance
    verify_tree_completeness,
    export_semantics,
    # Versioning
    ABSTENTION_TAXONOMY_VERSION,
    get_taxonomy_version,
)

__all__ = [
    # Abstention taxonomy
    "AbstentionType",
    "classify_verification_method",
    "classify_breakdown_key",
    "serialize_abstention",
    "deserialize_abstention",
    "ABSTENTION_METHOD_STRINGS",
    "is_abstention_method",
    # Failure classifier
    "FailureState",
    "classify_exception",
    "classify_and_log",
    "classify_from_result",
    "classify_from_status",
    "normalize_legacy_key",
    # Bridging functions
    "failure_to_abstention",
    "classify_breakdown_key_unified",
    # Abstention record
    "AbstentionRecord",
    "CANONICAL_ABSTENTION_ORDER",
    "CANONICAL_ABSTENTION_SET",
    "GLOBAL_VALIDATION_ENABLED",
    "create_canonical_histogram",
    "normalize_histogram",
    "merge_histograms",
    "histogram_to_records",
    # Semantic tree & categories
    "ABSTENTION_TREE",
    "SemanticCategory",
    "categorize",
    "get_category",
    "get_types_for_category",
    "get_all_categories",
    "is_timeout_related",
    "is_crash_related",
    "is_resource_related",
    "is_oracle_related",
    # Schema & validation
    "ABSTENTION_RECORD_SCHEMA",
    "get_schema",
    "get_schema_path",
    "AbstentionValidationError",
    "validate_abstention_data",
    "validate_abstention_record",
    "validate_abstention_json",
    # Aggregation
    "aggregate_by_category",
    "aggregate_histogram_by_category",
    # Analytics (Phase II v1.1)
    "summarize_abstentions",
    "detect_abstention_red_flags",
    # Phase III: Red-Flag Feed & Global Health
    "HEALTH_SNAPSHOT_SCHEMA_VERSION",
    "build_abstention_health_snapshot",
    "build_abstention_radar",
    "summarize_abstentions_for_uplift",
    "summarize_abstentions_for_global_health",
    # Phase IV: Epistemic Risk Decomposition & Cross-Signal Integration
    "EPISTEMIC_PROFILE_SCHEMA_VERSION",
    "build_epistemic_abstention_profile",
    "compose_abstention_with_budget_and_perf",
    "build_abstention_director_panel",
    "build_abstention_storyline",
    "evaluate_abstention_for_uplift",
    # Phase V: Double-Helix Drift Radar & Global Governance Linkage
    "DRIFT_TIMELINE_SCHEMA_VERSION",
    "build_epistemic_drift_timeline",
    "summarize_abstention_for_global_console",
    "compose_abstention_with_uplift_decision",
    # Governance
    "verify_tree_completeness",
    "export_semantics",
    # Versioning
    "ABSTENTION_TAXONOMY_VERSION",
    "get_taxonomy_version",
]
