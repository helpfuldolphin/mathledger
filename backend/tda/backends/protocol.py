"""
TDA Backend Protocol Definition.

Defines the abstract interface that all TDA backends must implement.
This allows swapping between Ripser, GUDHI, or fallback implementations.

References:
    - TDA_MIND_SCANNER_SPEC.md Section 7.1
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx
    from backend.tda.proof_complex import SimplicialComplex
    from backend.tda.metric_complex import TDAResult


class TDABackend(Protocol):
    """
    Protocol for TDA computation backends.

    Any backend must implement these methods to be usable by TDAMonitor.
    """

    def build_combinatorial_complex(
        self,
        local_dag: "nx.DiGraph",
        max_clique_size: int = 4,
    ) -> "SimplicialComplex":
        """
        Build a flag (clique) complex from a directed graph.

        Args:
            local_dag: NetworkX DiGraph representing proof structure
            max_clique_size: Maximum clique size to enumerate

        Returns:
            SimplicialComplex with the flag complex structure
        """
        ...

    def build_metric_complex(
        self,
        embeddings: Dict[str, np.ndarray],
        max_dim: int = 1,
        max_filtration: float | None = None,
    ) -> "TDAResult":
        """
        Build Vietoris-Rips complex and compute persistent homology.

        Args:
            embeddings: Dict mapping state IDs to feature vectors
            max_dim: Maximum homology dimension to compute
            max_filtration: Maximum filtration parameter

        Returns:
            TDAResult with persistence diagrams
        """
        ...

    def compute_betti(
        self,
        complex_: "SimplicialComplex",
        max_dim: int | None = None,
    ) -> Dict[int, int]:
        """
        Compute Betti numbers for a simplicial complex.

        Args:
            complex_: SimplicialComplex to analyze
            max_dim: Maximum dimension (default: complex's max_dim)

        Returns:
            Dict mapping dimension k to Betti number β_k
        """
        ...

    @property
    def name(self) -> str:
        """Backend name for logging/telemetry."""
        ...

    @property
    def is_available(self) -> bool:
        """Check if the backend's dependencies are available."""
        ...
