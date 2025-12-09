"""
TDA Runtime Monitor - Sidecar for U2Runner / RFL Integration.

This module implements the TDAMonitor class that integrates with MathLedger's
reasoning pipeline as specified in TDA_MIND_SCANNER_SPEC.md Section 4.

The monitor:
1. Accepts local proof DAGs and state embeddings per reasoning attempt
2. Builds combinatorial and metric complexes
3. Computes HSS and component scores
4. Emits gating signals (BLOCK / WARN / OK)

Integration pattern (Section 4.2):
    if tda_monitor.should_block(result):
        return ProofOutcome.ABANDONED_TDA
    if tda_monitor.should_warn(result):
        downweight_rfl_update()

References:
    - TDA_MIND_SCANNER_SPEC.md Section 4 (Architecture)
    - TDA_MIND_SCANNER_SPEC.md Section 5 (Operational Modes)
    - TDA_MIND_SCANNER_SPEC.md Section 6 (Safety Invariants)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx

logger = logging.getLogger(__name__)


class TDAGatingSignal(Enum):
    """
    Gating signal emitted by TDAMonitor.

    Per TDA_MIND_SCANNER_SPEC.md Section 3.7:
    - BLOCK: HSS < θ_block (structurally unstable)
    - WARN: θ_block ≤ HSS < θ_warn (marginal)
    - OK: HSS ≥ θ_warn (structurally coherent)
    """
    BLOCK = "BLOCK"
    WARN = "WARN"
    OK = "OK"


class TDAOperationalMode(Enum):
    """
    Operational mode for the TDA monitor.

    Per TDA_MIND_SCANNER_SPEC.md Section 5:
    - OFFLINE: No runtime effect, only analysis
    - SHADOW: Logging only, no gating
    - SOFT: Warn signals influence RFL/planner but no blocks
    - HARD: Block signals are hard constraints
    """
    OFFLINE = "offline"
    SHADOW = "shadow"
    SOFT = "soft"
    HARD = "hard"


class TDAMonitorError(Exception):
    """Exception raised by TDAMonitor operations."""
    pass


@dataclass
class TDAMonitorConfig:
    """
    Configuration for TDAMonitor.

    Per TDA_MIND_SCANNER_SPEC.md Section 4.3.1.

    Attributes:
        hss_block_threshold: θ_block for BLOCK signal (default: 0.2)
        hss_warn_threshold: θ_warn for WARN signal (default: 0.5)
        max_simplex_dim: Maximum simplex dimension for clique complex (default: 3)
        max_homology_dim: Maximum homology dimension to compute (default: 1)
        lifetime_threshold: τ for PCS computation (default: 0.05)
        deviation_max: δ_max for DRS normalization (default: 0.5)
        mode: Operational mode (default: SHADOW)
        fail_open: If True, TDA errors result in OK; if False, result in WARN (default: True)
        max_neighborhood_depth: Maximum depth for local DAG extraction (default: 3)
        max_embedding_window: Maximum states in embedding window (default: 200)
    """
    hss_block_threshold: float = 0.2
    hss_warn_threshold: float = 0.5
    max_simplex_dim: int = 3
    max_homology_dim: int = 1
    lifetime_threshold: float = 0.05
    deviation_max: float = 0.5
    mode: TDAOperationalMode = TDAOperationalMode.SHADOW
    fail_open: bool = True
    max_neighborhood_depth: int = 3
    max_embedding_window: int = 200

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.hss_block_threshold < 0 or self.hss_block_threshold > 1:
            raise ValueError("hss_block_threshold must be in [0, 1]")
        if self.hss_warn_threshold < 0 or self.hss_warn_threshold > 1:
            raise ValueError("hss_warn_threshold must be in [0, 1]")
        if self.hss_block_threshold >= self.hss_warn_threshold:
            raise ValueError("hss_block_threshold must be < hss_warn_threshold")
        if self.max_simplex_dim < 1:
            raise ValueError("max_simplex_dim must be >= 1")
        if self.max_homology_dim < 0:
            raise ValueError("max_homology_dim must be >= 0")


@dataclass
class TDAMonitorResult:
    """
    Result of a TDA monitoring evaluation.

    Per TDA_MIND_SCANNER_SPEC.md Section 4.3.2.

    Attributes:
        hss: Hallucination Stability Score [0, 1]
        sns: Structural Non-Triviality Score [0, 1]
        pcs: Persistence Coherence Score [0, 1]
        drs: Deviation-from-Reference Score [0, 1]
        signal: Gating signal (BLOCK, WARN, OK)
        block: True if signal is BLOCK
        warn: True if signal is WARN
        betti: Betti numbers from combinatorial complex
        metadata: Additional computation metadata
        computation_time_ms: Time taken to compute (milliseconds)
        error: Optional error message if computation failed
    """
    hss: float
    sns: float
    pcs: float
    drs: float
    signal: TDAGatingSignal
    block: bool
    warn: bool
    betti: Dict[int, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    computation_time_ms: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        """True if signal is OK."""
        return self.signal == TDAGatingSignal.OK

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for logging/telemetry."""
        return {
            "hss": self.hss,
            "sns": self.sns,
            "pcs": self.pcs,
            "drs": self.drs,
            "signal": self.signal.value,
            "block": self.block,
            "warn": self.warn,
            "betti": self.betti,
            "metadata": self.metadata,
            "computation_time_ms": self.computation_time_ms,
            "error": self.error,
        }


class TDAMonitor:
    """
    TDA Mind Scanner runtime monitor.

    Sidecar component that evaluates proof attempts for structural coherence
    using topological data analysis.

    Per TDA_MIND_SCANNER_SPEC.md Section 4.3.3.

    Usage:
        monitor = TDAMonitor(config, ref_profiles)
        result = monitor.evaluate_proof_attempt(
            slice_name="PL-1",
            local_dag=dag,
            embeddings=state_embeddings,
        )
        if monitor.should_block(result):
            # Prune branch / abstain
        elif monitor.should_warn(result):
            # Downweight RFL update
    """

    def __init__(
        self,
        config: TDAMonitorConfig,
        slice_ref_profiles: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize TDAMonitor.

        Args:
            config: TDAMonitorConfig with thresholds and parameters
            slice_ref_profiles: Dict mapping slice_name to ReferenceTDAProfile
        """
        config.validate()
        self.cfg = config
        self.slice_ref_profiles = slice_ref_profiles or {}

        # Import TDA modules
        from backend.tda.reference_profile import (
            ReferenceTDAProfile,
            get_or_create_profile,
        )
        self._ReferenceTDAProfile = ReferenceTDAProfile
        self._get_or_create_profile = get_or_create_profile

        # Statistics
        self._eval_count = 0
        self._block_count = 0
        self._warn_count = 0
        self._error_count = 0

    def evaluate_proof_attempt(
        self,
        slice_name: str,
        local_dag: "nx.DiGraph",
        embeddings: Dict[str, np.ndarray],
    ) -> TDAMonitorResult:
        """
        Evaluate a proof attempt for structural coherence.

        Per TDA_MIND_SCANNER_SPEC.md Section 4.3.3.

        Args:
            slice_name: Identifier for the current slice (e.g., "PL-1", "U2")
            local_dag: NetworkX DiGraph of the local proof structure
            embeddings: Dict mapping state IDs to feature vectors

        Returns:
            TDAMonitorResult with scores and gating signal
        """
        start_time = time.perf_counter()
        self._eval_count += 1

        try:
            result = self._evaluate_impl(slice_name, local_dag, embeddings)
        except Exception as e:
            logger.warning(f"TDA evaluation failed: {e}", exc_info=True)
            self._error_count += 1
            result = self._make_error_result(str(e))

        result.computation_time_ms = (time.perf_counter() - start_time) * 1000

        # Update statistics
        if result.block:
            self._block_count += 1
        elif result.warn:
            self._warn_count += 1

        return result

    def _evaluate_impl(
        self,
        slice_name: str,
        local_dag: "nx.DiGraph",
        embeddings: Dict[str, np.ndarray],
    ) -> TDAMonitorResult:
        """Internal implementation of evaluation."""
        from backend.tda.proof_complex import build_combinatorial_complex
        from backend.tda.metric_complex import build_metric_complex
        from backend.tda.scores import (
            compute_structural_nontriviality,
            compute_persistence_coherence,
            compute_deviation_from_reference,
            compute_hallucination_stability_score,
            ScoreWeights,
        )

        # Get reference profile for this slice
        ref_profile = self._get_or_create_profile(
            self.slice_ref_profiles, slice_name
        )

        # Build combinatorial complex from DAG
        comb_complex = build_combinatorial_complex(
            local_dag,
            max_clique_size=self.cfg.max_simplex_dim + 1,
        )

        # Build metric complex from embeddings
        # Limit to max_embedding_window states
        if len(embeddings) > self.cfg.max_embedding_window:
            # Take most recent states (assuming dict preserves order in Python 3.7+)
            keys = list(embeddings.keys())[-self.cfg.max_embedding_window:]
            embeddings = {k: embeddings[k] for k in keys}

        tda_result = build_metric_complex(
            embeddings,
            max_dim=self.cfg.max_homology_dim,
        )

        # Compute scores
        sns = compute_structural_nontriviality(comb_complex, ref_profile)
        pcs = compute_persistence_coherence(
            tda_result, ref_profile, self.cfg.lifetime_threshold
        )
        drs = compute_deviation_from_reference(
            tda_result, ref_profile, self.cfg.deviation_max
        )
        hss = compute_hallucination_stability_score(sns, pcs, drs)

        # Determine signal
        signal = self._classify_signal(hss)
        block = signal == TDAGatingSignal.BLOCK
        warn = signal == TDAGatingSignal.WARN

        # Apply mode constraints
        if self.cfg.mode == TDAOperationalMode.SHADOW:
            # Shadow mode: no gating, only logging
            block = False
            warn = False
        elif self.cfg.mode == TDAOperationalMode.SOFT:
            # Soft mode: no blocks, only warnings
            block = False

        # Extract Betti numbers
        betti = comb_complex.compute_betti_numbers(max_dim=self.cfg.max_homology_dim)

        # Build metadata
        metadata = {
            "slice": slice_name,
            "mode": self.cfg.mode.value,
            "num_nodes": comb_complex.num_vertices,
            "num_edges": comb_complex.num_edges,
            "num_simplices": comb_complex.num_simplices,
            "num_embeddings": len(embeddings),
            "tda_backend": tda_result.backend,
            "ref_profile_version": ref_profile.version,
        }

        return TDAMonitorResult(
            hss=hss,
            sns=sns,
            pcs=pcs,
            drs=drs,
            signal=signal,
            block=block,
            warn=warn,
            betti=betti,
            metadata=metadata,
        )

    def _classify_signal(self, hss: float) -> TDAGatingSignal:
        """Classify HSS into gating signal."""
        if hss < self.cfg.hss_block_threshold:
            return TDAGatingSignal.BLOCK
        elif hss < self.cfg.hss_warn_threshold:
            return TDAGatingSignal.WARN
        else:
            return TDAGatingSignal.OK

    def _make_error_result(self, error: str) -> TDAMonitorResult:
        """Create result for error cases."""
        if self.cfg.fail_open:
            # Fail-open: treat errors as OK
            signal = TDAGatingSignal.OK
            block = False
            warn = False
        else:
            # Fail-closed: treat errors as WARN
            signal = TDAGatingSignal.WARN
            block = False
            warn = True

        return TDAMonitorResult(
            hss=0.5,  # Neutral
            sns=0.0,
            pcs=0.0,
            drs=0.0,
            signal=signal,
            block=block,
            warn=warn,
            error=error,
            metadata={"error": True},
        )

    def should_block(self, result: TDAMonitorResult) -> bool:
        """
        Check if the result indicates blocking.

        Per TDA_MIND_SCANNER_SPEC.md Section 4.2:
        - In HARD mode, returns result.block
        - In other modes, always returns False
        """
        if self.cfg.mode != TDAOperationalMode.HARD:
            return False
        return result.block

    def should_warn(self, result: TDAMonitorResult) -> bool:
        """
        Check if the result indicates a warning.

        Returns True in SOFT and HARD modes when signal is WARN or BLOCK.
        """
        if self.cfg.mode in (TDAOperationalMode.OFFLINE, TDAOperationalMode.SHADOW):
            return False
        return result.warn or result.block

    def get_statistics(self) -> Dict[str, int]:
        """Get monitor statistics."""
        return {
            "eval_count": self._eval_count,
            "block_count": self._block_count,
            "warn_count": self._warn_count,
            "error_count": self._error_count,
        }

    def reset_statistics(self) -> None:
        """Reset monitor statistics."""
        self._eval_count = 0
        self._block_count = 0
        self._warn_count = 0
        self._error_count = 0


def create_monitor(
    mode: str = "shadow",
    block_threshold: float = 0.2,
    warn_threshold: float = 0.5,
    profiles_path: Optional[str] = None,
) -> TDAMonitor:
    """
    Factory function to create a TDAMonitor with common configurations.

    Args:
        mode: Operational mode ("offline", "shadow", "soft", "hard")
        block_threshold: HSS threshold for BLOCK
        warn_threshold: HSS threshold for WARN
        profiles_path: Optional path to reference profiles JSON

    Returns:
        Configured TDAMonitor instance
    """
    from pathlib import Path
    from backend.tda.reference_profile import load_reference_profiles

    config = TDAMonitorConfig(
        hss_block_threshold=block_threshold,
        hss_warn_threshold=warn_threshold,
        mode=TDAOperationalMode(mode),
    )

    profiles: Dict[str, Any] = {}
    if profiles_path:
        profiles = load_reference_profiles(Path(profiles_path))

    return TDAMonitor(config, profiles)
