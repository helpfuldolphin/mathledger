# PHASE II — NOT USED IN PHASE I
"""
Decoy Difficulty Scoring Engine for Phase II Uplift Experiments.

This package provides tools for measuring the "confusability" of decoy formulas
relative to their target counterparts. The scoring dimensions are:

1. Syntactic Proximity: Token/depth similarity
2. Atom Set Overlap: Jaccard similarity of atomic propositions
3. Structural Similarity: Connective patterns, implication chains
4. Semantic Confusability: Hash distance clustering

Higher scores indicate decoys that are harder to distinguish from targets.
"""

from .scoring import (
    DecoyScorer,
    DecoyScore,
    score_formula,
    score_slice_decoys,
    compute_confusability_index,
)

from .confusability import (
    compute_confusability,
    compute_confusability_components,
    ConfusabilityMap,
    ConfusabilityMapReport,
    FormulaConfusability,
    get_confusability_map,
    get_all_confusability_maps,
)

from .visualization import (
    plot_decoy_landscape,
    plot_all_landscapes,
    plot_comparison_landscape,
    generate_landscape_report,
)

from .contract import (
    SCHEMA_VERSION,
    FormulaEntry,
    ContractSummary,
    FamilyProfile,
    ConfusabilityContract,
    export_contract,
    validate_contract_schema,
    contracts_are_equal,
    compute_structure_fingerprint,
    get_difficulty_band,
)

from .risk import (
    RISK_SCHEMA_VERSION,
    RISK_LEVELS,
    RISK_ORDER,
    FORBIDDEN_WORDS,
    compute_family_risk_level,
    build_family_risk_snapshot,
    compare_family_risk,
    summarize_confusability_for_global_health,
    check_forbidden_language,
    build_slice_confusability_view,
    summarize_decoy_confusability_for_uplift,
    build_confusability_director_panel,
    build_decoy_family_drift_governor,
    build_decoy_uplift_prescreen,
    build_confusability_topology_coherence_map,
    build_confusability_drift_horizon_predictor,
    build_global_coherence_console_tile,
    COHERENCE_BAND_THRESHOLDS,
    COHERENCE_BAND_ORDER,
    FamilyRiskEntry,
    FamilyDriftResult,
    GovernanceSignal,
)

__all__ = [
    # Scoring
    "DecoyScorer",
    "DecoyScore",
    "score_formula",
    "score_slice_decoys",
    "compute_confusability_index",
    # Confusability
    "compute_confusability",
    "compute_confusability_components",
    "ConfusabilityMap",
    "ConfusabilityMapReport",
    "FormulaConfusability",
    "get_confusability_map",
    "get_all_confusability_maps",
    # Visualization
    "plot_decoy_landscape",
    "plot_all_landscapes",
    "plot_comparison_landscape",
    "generate_landscape_report",
    # Contract
    "SCHEMA_VERSION",
    "FormulaEntry",
    "ContractSummary",
    "FamilyProfile",
    "ConfusabilityContract",
    "export_contract",
    "validate_contract_schema",
    "contracts_are_equal",
    "compute_structure_fingerprint",
    "get_difficulty_band",
    # Risk
    "RISK_SCHEMA_VERSION",
    "RISK_LEVELS",
    "RISK_ORDER",
    "FORBIDDEN_WORDS",
    "compute_family_risk_level",
    "build_family_risk_snapshot",
    "compare_family_risk",
    "summarize_confusability_for_global_health",
    "check_forbidden_language",
    "build_slice_confusability_view",
    "summarize_decoy_confusability_for_uplift",
    "build_confusability_director_panel",
    "build_decoy_family_drift_governor",
    "build_decoy_uplift_prescreen",
    "build_confusability_topology_coherence_map",
    "build_confusability_drift_horizon_predictor",
    "build_global_coherence_console_tile",
    "COHERENCE_BAND_THRESHOLDS",
    "COHERENCE_BAND_ORDER",
    "FamilyRiskEntry",
    "FamilyDriftResult",
    "GovernanceSignal",
]

