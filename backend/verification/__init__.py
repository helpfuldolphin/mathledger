"""
Phase IIb Lean Verification Adapters.

STATUS: SCAFFOLD + SIMULATION — NOT USED IN RUNTIME YET.

This package provides the Lean adapter scaffold and related utilities
for Phase IIb uplift experiments. All Lean adapter functionality is
simulation-only until Phase IIb activation.
"""

# =============================================================================
# PHASE IIB LEAN ADAPTER EXPORTS (Always Available)
# =============================================================================
# The Lean adapter is a scaffold that defines types and interfaces for
# Phase IIb uplift experiments. It does NOT invoke Lean — all verify()
# calls return deterministic abstention results.

from backend.verification.lean_adapter import (
    # Constants
    LEAN_VERSION_REQUIRED,
    # Exceptions
    LeanAdapterValidationError,
    # Enums
    LeanAdapterMode,
    LeanAbstentionReason,
    VerificationErrorKind,
    # Dataclasses
    LeanResourceBudget,
    LeanVerificationRequest,
    LeanVerificationResult,
    LeanAdapterConfig,
    # Validation functions
    validate_verification_request,
    validate_resource_budget,
    is_valid_canonical,
    # Simulation helper
    simulate_lean_result,
    compute_formula_complexity,
    # Evidence Pack helper
    summarize_lean_activity,
    # Activity Ledger (Phase III)
    build_lean_activity_ledger,
    # Safety Envelope (Phase III)
    evaluate_lean_adapter_safety,
    # Global Health (Phase III)
    summarize_lean_for_global_health,
    # Capability Classification (Phase IV)
    classify_lean_capabilities,
    # Migration Checklist (Phase IV)
    build_lean_migration_checklist,
    # Director Panel (Phase IV)
    build_lean_director_panel,
    # Shadow Mode (Reality Bridge Protocol)
    generate_shadow_telemetry,
    build_lean_shadow_capability_radar,
    build_lean_director_panel_with_shadow,
    # Lean Mode Playbook (Next Mission)
    build_lean_mode_playbook,
    # Evidence Pack Adapter (Next Mission)
    summarize_lean_capabilities_for_evidence,
    # Main class
    LeanAdapter,
)

# =============================================================================
# PHASE IIB LEAN ADAPTER EXPORTS (always in __all__)
# =============================================================================
_LEAN_ADAPTER_EXPORTS = [
    # Constants
    "LEAN_VERSION_REQUIRED",
    # Exceptions
    "LeanAdapterValidationError",
    # Enums
    "LeanAdapterMode",
    "LeanAbstentionReason",
    "VerificationErrorKind",
    # Dataclasses
    "LeanResourceBudget",
    "LeanVerificationRequest",
    "LeanVerificationResult",
    "LeanAdapterConfig",
    # Validation functions
    "validate_verification_request",
    "validate_resource_budget",
    "is_valid_canonical",
    # Simulation helper
    "simulate_lean_result",
    "compute_formula_complexity",
    # Evidence Pack helper
    "summarize_lean_activity",
    # Activity Ledger (Phase III)
    "build_lean_activity_ledger",
    # Safety Envelope (Phase III)
    "evaluate_lean_adapter_safety",
    # Global Health (Phase III)
    "summarize_lean_for_global_health",
    # Capability Classification (Phase IV)
    "classify_lean_capabilities",
    # Migration Checklist (Phase IV)
    "build_lean_migration_checklist",
    # Director Panel (Phase IV)
    "build_lean_director_panel",
    # Shadow Mode (Reality Bridge Protocol)
    "generate_shadow_telemetry",
    "build_lean_shadow_capability_radar",
    "build_lean_director_panel_with_shadow",
    # Lean Mode Playbook (Next Mission)
    "build_lean_mode_playbook",
    # Evidence Pack Adapter (Next Mission)
    "summarize_lean_capabilities_for_evidence",
    # Main class
    "LeanAdapter",
]

# =============================================================================
# MOCK ORACLE EXPORTS (Conditional)
# =============================================================================
# Mock oracle components are conditionally exported based on environment.
# Some items (like SLICE_PROFILES) are always available from mock_config.
# Others (like MockVerifiableOracle) require the guard to be enabled.

# Always-available mock config exports
from backend.verification.mock_config import (
    # Contract version and definitions
    MOCK_ORACLE_CONTRACT_VERSION,
    PROFILE_CONTRACTS,
    NEGATIVE_CONTROL_CONTRACT,
    NegativeControlContract,
    # Contract verification
    verify_profile_contracts,
    verify_negative_control_result,
    # Contract export
    export_mock_oracle_contract,
    # Scenario layer
    Scenario,
    SCENARIOS,
    list_scenarios,
    get_scenario,
    # Scenario analytics
    summarize_scenario_results,
    detect_scenario_drift,
    # Governance functions
    SLICE_PROFILES,
    build_metric_scenario_coverage_view,
    summarize_mock_oracle_drift_for_governance,
    build_mock_oracle_director_panel,
    # Fleet console & regression watchdog
    build_scenario_fleet_summary,
    detect_mock_oracle_regression,
    # Coverage and result types
    ProfileCoverageMap,
    MockVerificationResult,
    compute_profile_coverage,
)

# Always-available mock exceptions
from backend.verification.mock_exceptions import (
    MockOracleCrashError,
    MockOracleTimeoutError,
    MockOracleError,
    MockOracleConfigError,
)

# Conditionally-available oracle implementation
try:
    from backend.verification.mock_oracle import MockVerifiableOracle
    from backend.verification.mock_expectations import MockOracleExpectations
    from backend.verification.mock_config import (
        MockOracleConfig,
    )
    
    _MOCK_ORACLE_EXPORTS = [
        "MockVerifiableOracle",
        "MockOracleExpectations",
        "MockOracleConfig",
    ]
except ImportError:
    _MOCK_ORACLE_EXPORTS = []

# Always-available mock config and exception exports
_MOCK_CONFIG_EXPORTS = [
    # Contract version and definitions
    "MOCK_ORACLE_CONTRACT_VERSION",
    "PROFILE_CONTRACTS",
    "NEGATIVE_CONTROL_CONTRACT",
    "NegativeControlContract",
    # Contract verification
    "verify_profile_contracts",
    "verify_negative_control_result",
    # Contract export
    "export_mock_oracle_contract",
    # Scenario layer
    "Scenario",
    "SCENARIOS",
    "list_scenarios",
    "get_scenario",
    # Scenario analytics
    "summarize_scenario_results",
    "detect_scenario_drift",
    # Governance functions
    "SLICE_PROFILES",
    "build_metric_scenario_coverage_view",
    "summarize_mock_oracle_drift_for_governance",
    "build_mock_oracle_director_panel",
    # Fleet console & regression watchdog
    "build_scenario_fleet_summary",
    "detect_mock_oracle_regression",
    # Coverage and result types
    "ProfileCoverageMap",
    "MockVerificationResult",
    "compute_profile_coverage",
    # Exceptions
    "MockOracleCrashError",
    "MockOracleTimeoutError",
    "MockOracleError",
    "MockOracleConfigError",
]

# =============================================================================
# PACKAGE EXPORTS
# =============================================================================

__all__ = _LEAN_ADAPTER_EXPORTS + _MOCK_ORACLE_EXPORTS + _MOCK_CONFIG_EXPORTS
