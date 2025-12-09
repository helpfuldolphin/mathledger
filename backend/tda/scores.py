"""
TDA Score Computation: SNS, PCS, DRS, HSS.

This module implements the four core scores from TDA_MIND_SCANNER_SPEC.md:

- SNS (Structural Non-Triviality Score): Section 3.3
  Composition of size factor and topology factor from Betti numbers.

- PCS (Persistence Coherence Score): Section 3.4
  Ratio of long-lived H_1 features to total lifetime mass.

- DRS (Deviation-from-Reference Score): Section 3.5
  Normalized bottleneck distance from healthy reference profile.

- HSS (Hallucination Stability Score): Section 3.6
  Weighted combination: HSS = clip((α·SNS + β·PCS - γ·DRS + γ) / (α+β+γ), 0, 1)

All functions are pure, deterministic, and unit-testable.

References:
    - TDA_MIND_SCANNER_SPEC.md Sections 3.3-3.6
    - Wasserman (2016), Sections 4-5
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.tda.proof_complex import SimplicialComplex
    from backend.tda.metric_complex import TDAResult, PersistenceDiagram
    from backend.tda.reference_profile import ReferenceTDAProfile


@dataclass
class ScoreWeights:
    """
    Weight configuration for HSS computation.

    Default values per TDA_MIND_SCANNER_SPEC.md Section 3.6:
    α = β = γ = 0.4

    Attributes:
        alpha: Weight for SNS (structural non-triviality)
        beta: Weight for PCS (persistence coherence)
        gamma: Weight for DRS (deviation penalty)
    """
    alpha: float = 0.4
    beta: float = 0.4
    gamma: float = 0.4

    def validate(self) -> None:
        """Ensure weights are non-negative."""
        if self.alpha < 0 or self.beta < 0 or self.gamma < 0:
            raise ValueError("All weights must be non-negative")

    @property
    def total(self) -> float:
        """Sum of all weights."""
        return self.alpha + self.beta + self.gamma


@dataclass
class ScoreResult:
    """
    Complete scoring result with all components.

    Attributes:
        sns: Structural Non-Triviality Score [0, 1]
        pcs: Persistence Coherence Score [0, 1]
        drs: Deviation-from-Reference Score [0, 1]
        hss: Hallucination Stability Score [0, 1]
        f_size: Size factor component of SNS
        f_topo: Topology factor component of SNS
        betti: Betti numbers used in computation
        weights: Weight configuration used
    """
    sns: float
    pcs: float
    drs: float
    hss: float
    f_size: float = 0.0
    f_topo: float = 0.0
    betti: Dict[int, int] = None  # type: ignore
    weights: ScoreWeights = None  # type: ignore

    def __post_init__(self) -> None:
        if self.betti is None:
            self.betti = {}
        if self.weights is None:
            self.weights = ScoreWeights()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "sns": self.sns,
            "pcs": self.pcs,
            "drs": self.drs,
            "hss": self.hss,
            "f_size": self.f_size,
            "f_topo": self.f_topo,
            "betti": self.betti,
            "weights": {
                "alpha": self.weights.alpha,
                "beta": self.weights.beta,
                "gamma": self.weights.gamma,
            },
        }


def compute_structural_nontriviality(
    complex_: "SimplicialComplex",
    ref_profile: Optional["ReferenceTDAProfile"] = None,
    n_ref: Optional[int] = None,
) -> float:
    """
    Compute Structural Non-Triviality Score (SNS).

    Per TDA_MIND_SCANNER_SPEC.md Section 3.3:

    SNS = f_size · f_topo

    Where:
    - f_size = min(1, log(1 + n_v) / log(1 + N_ref)) ∈ [0, 1]
    - f_topo is determined by Betti numbers:
        - 0    if β_0 > 1 (disconnected) and β_1 = 0
        - 0.5  if β_0 = 1 and β_1 = 0 (connected tree)
        - 1    if β_0 = 1 and β_1 > 0 (connected with cycles)
        - 0.25 otherwise

    Args:
        complex_: SimplicialComplex from proof DAG
        ref_profile: Optional reference profile containing N_ref
        n_ref: Override for reference node count (default: 50)

    Returns:
        SNS score in [0, 1]
    """
    # Get N_ref from profile or use default
    if n_ref is None:
        if ref_profile is not None and ref_profile.n_ref > 0:
            n_ref = ref_profile.n_ref
        else:
            n_ref = 50  # Default reference size

    # Number of vertices
    n_v = complex_.num_vertices

    # Compute size factor
    # f_size = min(1, log(1 + n_v) / log(1 + N_ref))
    if n_ref <= 0:
        f_size = 1.0 if n_v > 0 else 0.0
    else:
        log_nv = math.log(1 + n_v)
        log_nref = math.log(1 + n_ref)
        f_size = min(1.0, log_nv / log_nref) if log_nref > 0 else 0.0

    # Compute Betti numbers
    betti = complex_.compute_betti_numbers(max_dim=1)
    beta_0 = betti.get(0, 1)
    beta_1 = betti.get(1, 0)

    # Compute topology factor per spec
    f_topo = _compute_topology_factor(beta_0, beta_1)

    # SNS = f_size · f_topo
    sns = f_size * f_topo

    return sns


def compute_structural_nontriviality_detailed(
    complex_: "SimplicialComplex",
    ref_profile: Optional["ReferenceTDAProfile"] = None,
    n_ref: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute SNS with detailed breakdown.

    Returns dict with sns, f_size, f_topo, betti, n_v, n_ref.
    """
    if n_ref is None:
        if ref_profile is not None and ref_profile.n_ref > 0:
            n_ref = ref_profile.n_ref
        else:
            n_ref = 50

    n_v = complex_.num_vertices

    if n_ref <= 0:
        f_size = 1.0 if n_v > 0 else 0.0
    else:
        log_nv = math.log(1 + n_v)
        log_nref = math.log(1 + n_ref)
        f_size = min(1.0, log_nv / log_nref) if log_nref > 0 else 0.0

    betti = complex_.compute_betti_numbers(max_dim=1)
    beta_0 = betti.get(0, 1)
    beta_1 = betti.get(1, 0)

    f_topo = _compute_topology_factor(beta_0, beta_1)
    sns = f_size * f_topo

    return {
        "sns": sns,
        "f_size": f_size,
        "f_topo": f_topo,
        "betti": betti,
        "n_v": n_v,
        "n_ref": n_ref,
        "beta_0": beta_0,
        "beta_1": beta_1,
    }


def _compute_topology_factor(beta_0: int, beta_1: int) -> float:
    """
    Compute topology factor f_topo from Betti numbers.

    Per TDA_MIND_SCANNER_SPEC.md Section 3.3.2:
    - 0    if β_0 > 1 (disconnected) and β_1 = 0
    - 0.5  if β_0 = 1 and β_1 = 0 (connected tree)
    - 1    if β_0 = 1 and β_1 > 0 (connected with cycles)
    - 0.25 otherwise
    """
    if beta_0 > 1 and beta_1 == 0:
        # Disconnected and no loops: suspicious/trivial
        return 0.0
    elif beta_0 == 1 and beta_1 == 0:
        # Connected tree: minimally acceptable
        return 0.5
    elif beta_0 == 1 and beta_1 > 0:
        # Connected with cycles: structurally richer
        return 1.0
    else:
        # Edge cases (e.g., disconnected with loops)
        return 0.25


def compute_persistence_coherence(
    tda_result: "TDAResult",
    ref_profile: Optional["ReferenceTDAProfile"] = None,
    lifetime_threshold: Optional[float] = None,
    w1: float = 0.7,
    w0: float = 0.3,
) -> float:
    """
    Compute Persistence Coherence Score (PCS).

    Per TDA_MIND_SCANNER_SPEC.md Section 3.4:

    PCS = w_1 · PCS_1 + w_0 · PCS_0

    Where:
    - PCS_1 = L_long^{(1)} / L_total^{(1)} (ratio of long-lived H_1 features)
    - PCS_0 = analogous ratio for H_0
    - τ (lifetime_threshold) separates "long" from "short" features

    For v0, we primarily use PCS = PCS_1.

    Args:
        tda_result: TDAResult from metric complex computation
        ref_profile: Optional reference profile containing τ
        lifetime_threshold: Override for τ (default: 0.05)
        w1: Weight for H_1 contribution (default: 0.7)
        w0: Weight for H_0 contribution (default: 0.3)

    Returns:
        PCS score in [0, 1]
    """
    # Get lifetime threshold from profile or use default
    if lifetime_threshold is None:
        if ref_profile is not None and ref_profile.lifetime_threshold > 0:
            lifetime_threshold = ref_profile.lifetime_threshold
        else:
            lifetime_threshold = 0.05

    # Compute PCS_1 for H_1
    pcs_1 = _compute_pcs_for_dim(tda_result, dim=1, tau=lifetime_threshold)

    # Compute PCS_0 for H_0
    pcs_0 = _compute_pcs_for_dim(tda_result, dim=0, tau=lifetime_threshold)

    # Weighted combination
    pcs = w1 * pcs_1 + w0 * pcs_0

    return pcs


def _compute_pcs_for_dim(
    tda_result: "TDAResult",
    dim: int,
    tau: float,
) -> float:
    """
    Compute PCS for a single homology dimension.

    PCS_k = L_long^{(k)} / L_total^{(k)}

    Where:
    - L_total = sum of all finite lifetimes
    - L_long = sum of lifetimes > τ
    """
    diagram = tda_result.diagram(dim)
    lifetimes = diagram.lifetimes(exclude_essential=True)

    if not lifetimes:
        return 0.0

    l_total = sum(lifetimes)
    if l_total == 0:
        return 0.0

    l_long = sum(l for l in lifetimes if l > tau)

    return l_long / l_total


def compute_deviation_from_reference(
    tda_result: "TDAResult",
    ref_profile: Optional["ReferenceTDAProfile"],
    deviation_max: Optional[float] = None,
) -> float:
    """
    Compute Deviation-from-Reference Score (DRS).

    Per TDA_MIND_SCANNER_SPEC.md Section 3.5:

    DRS = min(1, d_B^{(1)} / δ_max)

    Where:
    - d_B^{(1)} = bottleneck distance between run's H_1 diagram and reference
    - δ_max = calibration constant (95th percentile of healthy distances)

    Args:
        tda_result: TDAResult from metric complex computation
        ref_profile: Reference profile containing D_ref^{(1)} and δ_max
        deviation_max: Override for δ_max (default: 0.5)

    Returns:
        DRS score in [0, 1]
    """
    # Get deviation_max from profile or use default
    if deviation_max is None:
        if ref_profile is not None and ref_profile.deviation_max > 0:
            deviation_max = ref_profile.deviation_max
        else:
            deviation_max = 0.5

    # If no reference profile, return neutral DRS
    if ref_profile is None:
        return 0.0

    # Import here to avoid circular dependency
    from backend.tda.metric_complex import bottleneck_distance

    # Get run's H_1 diagram
    run_diagram = tda_result.diagram(1)

    # Get reference H_1 diagram
    ref_diagram = ref_profile.reference_diagram_h1

    if ref_diagram is None:
        return 0.0

    # Compute bottleneck distance
    d_b = bottleneck_distance(run_diagram, ref_diagram)

    # Normalize by deviation_max
    drs = min(1.0, d_b / deviation_max) if deviation_max > 0 else 0.0

    return drs


def compute_hallucination_stability_score(
    sns: float,
    pcs: float,
    drs: float,
    weights: Optional[ScoreWeights] = None,
) -> float:
    """
    Compute Hallucination Stability Score (HSS).

    Per TDA_MIND_SCANNER_SPEC.md Section 3.6:

    raw = α · SNS + β · PCS - γ · DRS
    HSS = clip((raw + γ) / (α + β + γ), 0, 1)

    This maps the range [-γ, α+β] to approximately [0, 1].

    Args:
        sns: Structural Non-Triviality Score [0, 1]
        pcs: Persistence Coherence Score [0, 1]
        drs: Deviation-from-Reference Score [0, 1]
        weights: ScoreWeights (default: α=β=γ=0.4)

    Returns:
        HSS score in [0, 1]
    """
    if weights is None:
        weights = ScoreWeights()

    weights.validate()

    alpha = weights.alpha
    beta = weights.beta
    gamma = weights.gamma

    # raw = α · SNS + β · PCS - γ · DRS
    raw = alpha * sns + beta * pcs - gamma * drs

    # Normalize and clamp
    total = alpha + beta + gamma
    if total == 0:
        return 0.5  # Neutral if all weights are zero

    # HSS = clip((raw + γ) / (α + β + γ), 0, 1)
    hss = (raw + gamma) / total
    hss = max(0.0, min(1.0, hss))

    return hss


def compute_all_scores(
    complex_: "SimplicialComplex",
    tda_result: "TDAResult",
    ref_profile: Optional["ReferenceTDAProfile"] = None,
    weights: Optional[ScoreWeights] = None,
) -> ScoreResult:
    """
    Compute all TDA scores in one call.

    Args:
        complex_: SimplicialComplex from proof DAG
        tda_result: TDAResult from metric complex
        ref_profile: Optional reference profile
        weights: Optional weight configuration

    Returns:
        ScoreResult with all scores and metadata
    """
    if weights is None:
        weights = ScoreWeights()

    # Compute SNS with details
    sns_details = compute_structural_nontriviality_detailed(
        complex_, ref_profile
    )
    sns = sns_details["sns"]

    # Compute PCS
    pcs = compute_persistence_coherence(tda_result, ref_profile)

    # Compute DRS
    drs = compute_deviation_from_reference(tda_result, ref_profile)

    # Compute HSS
    hss = compute_hallucination_stability_score(sns, pcs, drs, weights)

    return ScoreResult(
        sns=sns,
        pcs=pcs,
        drs=drs,
        hss=hss,
        f_size=sns_details["f_size"],
        f_topo=sns_details["f_topo"],
        betti=sns_details["betti"],
        weights=weights,
    )


def classify_hss(
    hss: float,
    block_threshold: float = 0.2,
    warn_threshold: float = 0.5,
) -> str:
    """
    Classify HSS into operational categories.

    Per TDA_MIND_SCANNER_SPEC.md Section 3.7:
    - HSS < θ_block: BLOCK
    - θ_block ≤ HSS < θ_warn: WARN
    - HSS ≥ θ_warn: OK

    Args:
        hss: Hallucination Stability Score
        block_threshold: θ_block (default: 0.2)
        warn_threshold: θ_warn (default: 0.5)

    Returns:
        Classification string: "BLOCK", "WARN", or "OK"
    """
    if hss < block_threshold:
        return "BLOCK"
    elif hss < warn_threshold:
        return "WARN"
    else:
        return "OK"
