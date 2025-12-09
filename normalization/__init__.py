from .canon import normalize, normalize_pretty, are_equivalent, get_atomic_propositions, canonical_bytes
from .taut import (
    truth_table_is_tautology,
    TruthTableTimeout,
    clear_oracle_cache,
)
from .truthtab import is_tautology
from .tt_chi import (
    chi_estimate,
    chi_from_diagnostics,
    CHIResult,
    estimate_timeout_budget,
    format_chi_report,
    format_chi_compact,
    classify_hardness,
    get_hardness_description,
    BASELINE_NS_PER_ASSIGNMENT,
    HARDNESS_THRESHOLDS,
    # Phase II UX & Policy Layer (v1.1)
    suggest_timeout_ms,
    HardnessPolicySignal,
    format_diagnostics_for_report,
    TIMEOUT_HINTS_BY_CATEGORY,
    # Phase III Risk Envelope & Policy Hooks (v1.2)
    build_hardness_risk_envelope,
    derive_timeout_policy_recommendation,
    summarize_tt_hardness_for_global_health,
    RISK_ENVELOPE_SCHEMA_VERSION,
    RISK_BAND_BY_CATEGORY,
    RISK_BAND_NOTES,
    POLICY_HINT_BY_RISK_BAND,
    # Phase IV Hardness-Aware Workload Shaping & Slice Policy Feeds (v1.3)
    build_slice_hardness_profile,
    derive_tt_workload_shaping_policy,
    summarize_slice_hardness_for_curriculum,
    summarize_tt_risk_for_global_health,
    SLICE_PROFILE_SCHEMA_VERSION,
    WORKLOAD_SHAPING_HINTS,
    CURRICULUM_HINTS,
    # Phase IV Extension: Curriculum Gate & TT Capacity Tile
    evaluate_slice_hardness_for_curriculum,
    summarize_tt_capacity_for_global_health,
    DEFAULT_CURRICULUM_GATE_CONFIG,
)
