"""
TDA Mind Scanner - Topological Data Analysis for MathLedger reasoning processes.

This module implements Operation CORTEX: a topology-driven runtime monitor that
observes proof DAGs and reasoning trajectories, builds topological summaries,
and computes the Hallucination Stability Score (HSS) to gate reasoning processes.

Architecture:
    - proof_complex: Combinatorial (flag/clique) complex construction from proof DAGs
    - metric_complex: Vietoris-Rips filtration and persistent homology from embeddings
    - scores: SNS, PCS, DRS, HSS computation per TDA_MIND_SCANNER_SPEC.md
    - reference_profile: Calibration profiles for healthy reasoning per slice
    - runtime_monitor: TDAMonitor sidecar for U2Runner/RFL integration
    - backends: Ripser/GUDHI abstractions for persistent homology

References:
    - docs/TDA_MIND_SCANNER_SPEC.md (canonical specification)
    - Wasserman, L. (2016). Topological Data Analysis.

Phase: V (CORTEX)
Status: Implementation
"""

from backend.tda.proof_complex import (
    SimplicialComplex,
    build_combinatorial_complex,
    extract_local_neighborhood,
)
from backend.tda.metric_complex import (
    TDAResult,
    PersistenceDiagram,
    build_metric_complex,
)
from backend.tda.scores import (
    compute_structural_nontriviality,
    compute_persistence_coherence,
    compute_deviation_from_reference,
    compute_hallucination_stability_score,
)
from backend.tda.reference_profile import (
    ReferenceTDAProfile,
    build_reference_profile,
    load_reference_profiles,
    save_reference_profiles,
)
from backend.tda.runtime_monitor import (
    TDAMonitor,
    TDAMonitorConfig,
    TDAMonitorResult,
    TDAGatingSignal,
    TDAMonitorError,
)

__all__ = [
    # proof_complex
    "SimplicialComplex",
    "build_combinatorial_complex",
    "extract_local_neighborhood",
    # metric_complex
    "TDAResult",
    "PersistenceDiagram",
    "build_metric_complex",
    # scores
    "compute_structural_nontriviality",
    "compute_persistence_coherence",
    "compute_deviation_from_reference",
    "compute_hallucination_stability_score",
    # reference_profile
    "ReferenceTDAProfile",
    "build_reference_profile",
    "load_reference_profiles",
    "save_reference_profiles",
    # runtime_monitor
    "TDAMonitor",
    "TDAMonitorConfig",
    "TDAMonitorResult",
    "TDAGatingSignal",
    "TDAMonitorError",
]

__version__ = "0.1.0"
__spec_version__ = "TDA_MIND_SCANNER_SPEC v0.1"
