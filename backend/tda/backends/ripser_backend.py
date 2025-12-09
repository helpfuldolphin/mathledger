"""
Ripser Backend for TDA Computation.

Uses the ripser library for fast persistent homology computation.
Ripser is optimized for Vietoris-Rips filtrations and is the recommended
backend for production use.

References:
    - https://ripser.scikit-tda.org/
    - TDA_MIND_SCANNER_SPEC.md Section 7.1
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx
    from backend.tda.proof_complex import SimplicialComplex
    from backend.tda.metric_complex import TDAResult

# Check ripser availability
try:
    import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    ripser = None  # type: ignore


class RipserBackend:
    """
    TDA backend using the ripser library.

    Ripser is a lean C++ code for the computation of Vietoris-Rips persistence
    barcodes. It is extremely fast for moderate-sized point clouds.

    Usage:
        backend = RipserBackend()
        if backend.is_available:
            result = backend.build_metric_complex(embeddings)
    """

    @property
    def name(self) -> str:
        """Backend name."""
        return "ripser"

    @property
    def is_available(self) -> bool:
        """Check if ripser is installed."""
        return HAS_RIPSER

    def build_combinatorial_complex(
        self,
        local_dag: "nx.DiGraph",
        max_clique_size: int = 4,
    ) -> "SimplicialComplex":
        """
        Build combinatorial complex using shared implementation.

        Ripser is for metric complexes; combinatorial uses networkx.
        """
        from backend.tda.proof_complex import build_combinatorial_complex
        return build_combinatorial_complex(local_dag, max_clique_size)

    def build_metric_complex(
        self,
        embeddings: Dict[str, np.ndarray],
        max_dim: int = 1,
        max_filtration: Optional[float] = None,
    ) -> "TDAResult":
        """
        Build Vietoris-Rips complex using ripser.

        Args:
            embeddings: Dict mapping state IDs to feature vectors
            max_dim: Maximum homology dimension (default: 1)
            max_filtration: Maximum filtration value (default: auto)

        Returns:
            TDAResult with persistence diagrams
        """
        if not HAS_RIPSER:
            raise ImportError("ripser is not installed")

        from backend.tda.metric_complex import (
            TDAResult,
            PersistenceDiagram,
            PersistenceInterval,
        )

        # Convert embeddings to point cloud
        if not embeddings:
            return self._empty_result(max_dim)

        points = np.array(list(embeddings.values()), dtype=np.float64)

        if points.ndim == 1:
            points = points.reshape(-1, 1)

        if points.shape[0] < 2:
            return self._empty_result(max_dim, num_points=points.shape[0])

        # Compute max filtration if not provided
        if max_filtration is None:
            from scipy.spatial.distance import pdist
            distances = pdist(points)
            max_filtration = float(np.max(distances)) if len(distances) > 0 else 1.0

        # Run ripser
        result = ripser.ripser(
            points,
            maxdim=max_dim,
            thresh=max_filtration,
        )

        # Convert to TDAResult
        diagrams: Dict[int, PersistenceDiagram] = {}
        for dim, dgm in enumerate(result['dgms']):
            intervals = []
            for birth, death in dgm:
                intervals.append(PersistenceInterval(
                    birth=float(birth),
                    death=float(death),
                    dimension=dim,
                ))
            diagrams[dim] = PersistenceDiagram(dimension=dim, intervals=intervals)

        return TDAResult(
            diagrams=diagrams,
            num_points=points.shape[0],
            embedding_dim=points.shape[1],
            max_homology_dim=max_dim,
            backend="ripser",
            max_filtration=max_filtration,
        )

    def compute_betti(
        self,
        complex_: "SimplicialComplex",
        max_dim: Optional[int] = None,
    ) -> Dict[int, int]:
        """Compute Betti numbers using complex's method."""
        return complex_.compute_betti_numbers(max_dim)

    def _empty_result(
        self,
        max_dim: int,
        num_points: int = 0,
    ) -> "TDAResult":
        """Create empty result for edge cases."""
        from backend.tda.metric_complex import TDAResult, PersistenceDiagram

        diagrams = {k: PersistenceDiagram(dimension=k) for k in range(max_dim + 1)}
        return TDAResult(
            diagrams=diagrams,
            num_points=num_points,
            embedding_dim=0,
            max_homology_dim=max_dim,
            backend="ripser-empty",
        )


def get_ripser_backend() -> Optional[RipserBackend]:
    """
    Get a RipserBackend instance if available.

    Returns:
        RipserBackend if ripser is installed, None otherwise
    """
    backend = RipserBackend()
    return backend if backend.is_available else None
