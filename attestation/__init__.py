"""MathLedger dual attestation package."""

from .dual_root import (
    compute_composite_root,
    compute_reasoning_root,
    compute_ui_root,
    generate_attestation_metadata,
)
from .tda_pipeline import (
    TDAPipelineConfig,
    compute_tda_pipeline_hash,
    detect_tda_divergence,
)
from .chain_verifier import (
    AttestationVerificationError,
    ExperimentBlock,
    AttestationChainVerifier,
    verify_experiment_attestation_chain,
)

__all__: list[str] = [
    "compute_composite_root",
    "compute_reasoning_root",
    "compute_ui_root",
    "generate_attestation_metadata",
    "TDAPipelineConfig",
    "compute_tda_pipeline_hash",
    "detect_tda_divergence",
    "AttestationVerificationError",
    "ExperimentBlock",
    "AttestationChainVerifier",
    "verify_experiment_attestation_chain",
]

