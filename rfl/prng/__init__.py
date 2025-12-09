# PHASE II — NOT USED IN PHASE I
"""
Deterministic PRNG module for Phase II U2 experiments.

This module implements a hierarchical, SHA-256-based stateless PRNG
for fully reproducible randomness in uplift experiments.

Usage:
    from rfl.prng import DeterministicPRNG, PRNGKey, derive_seed

    prng = DeterministicPRNG("0" * 64)  # 64-hex master seed
    rng = prng.for_path("slice_uplift_sparse", "baseline", "cycle_0001", "ordering")
    shuffled = rng.shuffle(items)

Contract Reference:
    This module implements the deterministic PRNG contract described in
    docs/DETERMINISM_CONTRACT.md - specifically the hierarchical seed
    derivation chain required for Phase II experiments.

Integrity Guard:
    To enable runtime detection of global randomness violations:
        export RFL_PRNG_GUARD=1    # Auto-install guards on import
        export RFL_PRNG_STRICT=1   # Raise on violation
        export RFL_PRNG_WARN=1     # Warn on violation
"""

from .deterministic_prng import (
    PRNGKey,
    derive_seed,
    DeterministicPRNG,
    int_to_hex_seed,
    DEFAULT_MASTER_SEED,
)

from .integrity_guard import (
    install_guards,
    uninstall_guards,
    get_violation_log,
    clear_violation_log,
    check_module_compliance,
    audit_phase_ii_modules,
)

from .governance import (
    GovernanceStatus,
    ManifestStatus,
    PRNG_GOV_RULES,
    PRNGGovernanceSnapshot,
    PolicyEvaluation,
    GlobalHealthSummary,
    build_prng_governance_snapshot,
    evaluate_prng_policy,
    summarize_prng_for_global_health,
    run_full_prng_governance,
    evaluate_prng_for_ci,
    build_prng_remediation_suggestions,
    build_prng_governance_history,
)

__all__ = [
    # Core PRNG
    "PRNGKey",
    "derive_seed",
    "DeterministicPRNG",
    "int_to_hex_seed",
    "DEFAULT_MASTER_SEED",
    # Integrity Guard
    "install_guards",
    "uninstall_guards",
    "get_violation_log",
    "clear_violation_log",
    "check_module_compliance",
    "audit_phase_ii_modules",
    # Governance
    "GovernanceStatus",
    "ManifestStatus",
    "PRNG_GOV_RULES",
    "PRNGGovernanceSnapshot",
    "PolicyEvaluation",
    "GlobalHealthSummary",
    "build_prng_governance_snapshot",
    "evaluate_prng_policy",
    "summarize_prng_for_global_health",
    "run_full_prng_governance",
    # Phase IV
    "evaluate_prng_for_ci",
    "build_prng_remediation_suggestions",
    "build_prng_governance_history",
]

