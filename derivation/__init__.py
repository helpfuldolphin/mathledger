"""MathLedger derivation engine package."""

from .bounds import SliceBounds
from .pipeline import (
    ABSTENTION_METHODS,
    TELEMETRY_VERSION,
    AbstainedStatement,
    ASTNormalizationConfig,
    DEFAULT_AST_CONFIG,
    DerivationOutcome,
    DerivationPipeline,
    DerivationResult,
    DerivationSummary,
    FirstOrganismDerivationConfig,
    NormalizationStrategy,
    PipelineStats,
    StatementRecord,
    make_first_organism_derivation_config,
    make_first_organism_derivation_slice,
    make_first_organism_seed_statements,
    run_slice_for_test,
)
from .verification import StatementVerifier, VerificationOutcome
from .budget_invariants import (
    BUDGET_SCHEMA_VERSION,
    build_budget_episode_ledger_tile,
    build_budget_invariant_snapshot,
    build_budget_invariant_timeline,
    build_budget_invariants_director_panel,
    build_budget_invariants_governance_view,
    build_budget_storyline,
    evaluate_budget_release_readiness,
    explain_budget_release_decision,
    project_budget_stability_horizon,
    summarize_budget_invariants_for_global_health,
    summarize_storyline_for_global_health,
)

__all__: list[str] = [
    # Constants
    "ABSTENTION_METHODS",
    "TELEMETRY_VERSION",
    
    # Core types
    "AbstainedStatement",
    "DerivationOutcome",
    "DerivationPipeline",
    "DerivationResult",
    "DerivationSummary",
    "FirstOrganismDerivationConfig",
    "PipelineStats",
    "SliceBounds",
    "StatementRecord",
    "StatementVerifier",
    "VerificationOutcome",
    
    # First Organism
    "make_first_organism_derivation_config",
    "make_first_organism_derivation_slice",
    "make_first_organism_seed_statements",
    "run_slice_for_test",
    
    # AST normalization (future FOL)
    "ASTNormalizationConfig",
    "DEFAULT_AST_CONFIG",
    "NormalizationStrategy",
    
    # Budget invariants and governance
    "BUDGET_SCHEMA_VERSION",
    "build_budget_episode_ledger_tile",
    "build_budget_invariant_snapshot",
    "build_budget_invariant_timeline",
    "build_budget_invariants_director_panel",
    "build_budget_invariants_governance_view",
    "build_budget_storyline",
    "evaluate_budget_release_readiness",
    "explain_budget_release_decision",
    "project_budget_stability_horizon",
    "summarize_budget_invariants_for_global_health",
    "summarize_storyline_for_global_health",
    "attach_budget_invariants_to_evidence",
    "build_first_light_budget_storyline",
]
from .budget_cal_exp_integration import (
    annotate_cal_exp_windows_with_budget_storyline,
    attach_budget_storyline_to_cal_exp_report,
    build_budget_confounding_truth_table,
    build_cal_exp_budget_storyline_from_snapshots,
    extract_budget_storyline_from_cal_exp_report,
    validate_budget_confounding_defaults,
)

__all__.extend([
    "annotate_cal_exp_windows_with_budget_storyline",
    "attach_budget_storyline_to_cal_exp_report",
    "build_budget_confounding_truth_table",
    "build_cal_exp_budget_storyline_from_snapshots",
    "extract_budget_storyline_from_cal_exp_report",
    "validate_budget_confounding_defaults",
])
