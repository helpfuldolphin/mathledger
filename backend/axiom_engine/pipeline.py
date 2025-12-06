"""Deprecated pipeline shim — use derivation.pipeline instead."""

import warnings

from derivation.pipeline import (
    AbstainedStatement,
    DerivationOutcome,
    DerivationPipeline,
    DerivationResult,
    DerivationSummary,
    FirstOrganismDerivationConfig,
    PipelineStats,
    StatementRecord,
    make_first_organism_derivation_config,
    run_slice_for_test,
)

warnings.warn(
    "backend.axiom_engine.pipeline is deprecated; import derivation.pipeline instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "AbstainedStatement",
    "DerivationOutcome",
    "DerivationPipeline",
    "DerivationResult",
    "DerivationSummary",
    "FirstOrganismDerivationConfig",
    "PipelineStats",
    "StatementRecord",
    "make_first_organism_derivation_config",
    "run_slice_for_test",
]
