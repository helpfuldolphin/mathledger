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
    SCHEMA_VERSION as BUDGET_SCHEMA_VERSION,
    build_budget_invariant_snapshot,
    build_budget_invariant_timeline,
    summarize_budget_invariants_for_global_health,
    build_budget_invariants_governance_view,
    evaluate_budget_release_readiness,
    build_budget_invariants_director_panel,
    build_budget_storyline,
    explain_budget_release_decision,
)
try:  # Legacy compatibility: DEBUG_BUDGET_ENABLED was removed in Phase II.
    from .pipeline import summarize_budget, DEBUG_BUDGET_ENABLED
except ImportError:
    from .pipeline import summarize_budget  # type: ignore
    DEBUG_BUDGET_ENABLED = False  # pragma: no cover

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
    
    # Budget utilities (Phase II)
    "DEBUG_BUDGET_ENABLED",
    "summarize_budget",
    
    # Budget invariant governance (Phase III)
    "BUDGET_SCHEMA_VERSION",
    "build_budget_invariant_snapshot",
    "build_budget_invariant_timeline",
    "summarize_budget_invariants_for_global_health",
    # Phase IV cross-layer governance
    "build_budget_invariants_governance_view",
    "evaluate_budget_release_readiness",
    "build_budget_invariants_director_panel",
    # Phase V narrative and forensics
    "build_budget_storyline",
    "explain_budget_release_decision",
]
