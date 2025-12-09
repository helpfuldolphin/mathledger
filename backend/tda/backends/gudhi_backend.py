"""
GUDHI Backend for TDA Computation.

Uses the GUDHI library for persistent homology computation.
GUDHI provides more features than Ripser (alpha complexes, cubical complexes)
but may be slower for simple Vietoris-Rips computations.

References:
    - https://gudhi.inria.fr/
    - TDA_MIND_SCANNER_SPEC.md Section 7.1
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx
    from backend.tda.proof_complex import SimplicialComplex
    from backend.tda.metric_complex import TDAResult

# Check GUDHI availability
try:
    import gudhi
    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False
    gudhi = None  # type: ignore


class GUDHIBackend:
    """
    TDA backend using the GUDHI library.

    GUDHI (Geometry Understanding in Higher Dimensions) provides
    comprehensive TDA functionality including multiple complex types.

    Usage:
        backend = GUDHIBackend()
        if backend.is_available:
            result = backend.build_metric_complex(embeddings)
    """

    @property
    def name(self) -> str:
        """Backend name."""
        return "gudhi"

    @property
    def is_available(self) -> bool:
        """Check if GUDHI is installed."""
        return HAS_GUDHI

    def build_combinatorial_complex(
        self,
        local_dag: "nx.DiGraph",
        max_clique_size: int = 4,
    ) -> "SimplicialComplex":
        """
        Build combinatorial complex using shared implementation.

        GUDHI could use SimplexTree here, but networkx approach is simpler.
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
        Build Vietoris-Rips complex using GUDHI.

        Args:
            embeddings: Dict mapping state IDs to feature vectors
            max_dim: Maximum homology dimension (default: 1)
            max_filtration: Maximum edge length (default: auto)

        Returns:
            TDAResult with persistence diagrams
        """
        if not HAS_GUDHI:
            raise ImportError("gudhi is not installed")

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

        # Build Rips complex using GUDHI
        rips = gudhi.RipsComplex(
            points=points.tolist(),
            max_edge_length=max_filtration,
        )

        # Create simplex tree with dimension limit
        simplex_tree = rips.create_simplex_tree(max_dimension=max_dim + 1)

        # Compute persistence
        simplex_tree.compute_persistence()

        # Extract persistence diagrams
        diagrams: Dict[int, PersistenceDiagram] = {}
        for dim in range(max_dim + 1):
            intervals = []
            for interval in simplex_tree.persistence_intervals_in_dimension(dim):
                birth, death = interval
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
            backend="gudhi",
            max_filtration=max_filtration,
        )

    def build_alpha_complex(
        self,
        embeddings: Dict[str, np.ndarray],
        max_dim: int = 1,
    ) -> "TDAResult":
        """
        Build Alpha complex using GUDHI.

        Alpha complexes are often more efficient than Rips for low dimensions.
        Only available with GUDHI backend.

        Args:
            embeddings: Dict mapping state IDs to feature vectors
            max_dim: Maximum homology dimension

        Returns:
            TDAResult with persistence diagrams
        """
        if not HAS_GUDHI:
            raise ImportError("gudhi is not installed")

        from backend.tda.metric_complex import (
            TDAResult,
            PersistenceDiagram,
            PersistenceInterval,
        )

        if not embeddings:
            return self._empty_result(max_dim)

        points = np.array(list(embeddings.values()), dtype=np.float64)

        if points.ndim == 1:
            points = points.reshape(-1, 1)

        if points.shape[0] < 2:
            return self._empty_result(max_dim, num_points=points.shape[0])

        # Build Alpha complex
        alpha = gudhi.AlphaComplex(points=points.tolist())
        simplex_tree = alpha.create_simplex_tree()

        # Compute persistence
        simplex_tree.compute_persistence()

        # Extract diagrams
        diagrams: Dict[int, PersistenceDiagram] = {}
        for dim in range(max_dim + 1):
            intervals = []
            for interval in simplex_tree.persistence_intervals_in_dimension(dim):
                birth, death = interval
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
            backend="gudhi-alpha",
            max_filtration=float('inf'),
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
            backend="gudhi-empty",
        )


def get_gudhi_backend() -> Optional[GUDHIBackend]:
    """
    Get a GUDHIBackend instance if available.

    Returns:
        GUDHIBackend if GUDHI is installed, None otherwise
    """
    backend = GUDHIBackend()
    return backend if backend.is_available else None
