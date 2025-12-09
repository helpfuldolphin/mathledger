"""
Metric Complex Construction via Vietoris-Rips Filtration.

This module implements the mapping from state embeddings to persistent homology
as specified in TDA_MIND_SCANNER_SPEC.md Section 3.2.

Given embeddings X = {φ(s_t)} ⊂ ℝ^d:
1. Build Vietoris-Rips complex at increasing scale parameters ε
2. Compute persistent homology across the filtration
3. Extract persistence diagrams D^{(k)} for dimensions k = 0, 1, ...

The persistence diagrams capture:
- H_0: cluster structure (connected components)
- H_1: loops/cycles in the state manifold (oscillatory reasoning)
- Higher: voids and cavities (optional)

References:
    - TDA_MIND_SCANNER_SPEC.md Section 3.2
    - Wasserman (2016), Section 5 (Persistent Homology)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Backend detection
try:
    import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    ripser = None  # type: ignore

try:
    import gudhi
    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False
    gudhi = None  # type: ignore

try:
    from persim import bottleneck
    HAS_PERSIM = True
except ImportError:
    HAS_PERSIM = False
    bottleneck = None  # type: ignore


@dataclass
class PersistenceInterval:
    """
    A single persistence interval (birth, death) for a homology class.

    Attributes:
        birth: Scale parameter at which the feature is born
        death: Scale parameter at which the feature dies (inf for essential features)
        dimension: Homology dimension (0 = component, 1 = loop, etc.)
    """
    birth: float
    death: float
    dimension: int

    @property
    def lifetime(self) -> float:
        """Compute the persistence lifetime ℓ = death - birth."""
        if np.isinf(self.death):
            return float('inf')
        return self.death - self.birth

    @property
    def is_essential(self) -> bool:
        """Check if this is an essential feature (infinite lifetime)."""
        return np.isinf(self.death)

    def to_tuple(self) -> Tuple[float, float]:
        """Return (birth, death) tuple."""
        return (self.birth, self.death)


@dataclass
class PersistenceDiagram:
    """
    Persistence diagram for a single homology dimension.

    A multiset of (birth, death) pairs representing the birth and death
    times of homology classes across the filtration.

    Attributes:
        dimension: Homology dimension (0, 1, 2, ...)
        intervals: List of PersistenceInterval objects
    """
    dimension: int
    intervals: List[PersistenceInterval] = field(default_factory=list)

    @property
    def num_intervals(self) -> int:
        """Number of intervals in the diagram."""
        return len(self.intervals)

    @property
    def total_lifetime(self) -> float:
        """Sum of all finite lifetimes."""
        return sum(
            iv.lifetime for iv in self.intervals
            if not np.isinf(iv.lifetime)
        )

    @property
    def max_lifetime(self) -> float:
        """Maximum finite lifetime."""
        finite_lifetimes = [
            iv.lifetime for iv in self.intervals
            if not np.isinf(iv.lifetime)
        ]
        return max(finite_lifetimes) if finite_lifetimes else 0.0

    def lifetimes(self, exclude_essential: bool = True) -> List[float]:
        """
        Get list of lifetimes.

        Args:
            exclude_essential: If True, exclude infinite lifetimes
        """
        if exclude_essential:
            return [
                iv.lifetime for iv in self.intervals
                if not np.isinf(iv.lifetime)
            ]
        return [iv.lifetime for iv in self.intervals]

    def to_numpy(self) -> np.ndarray:
        """
        Convert to numpy array of shape (n, 2) for (birth, death) pairs.

        Essential features (infinite death) are represented with death = np.inf.
        """
        if not self.intervals:
            return np.zeros((0, 2), dtype=np.float64)

        return np.array(
            [iv.to_tuple() for iv in self.intervals],
            dtype=np.float64
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "dimension": self.dimension,
            "num_intervals": self.num_intervals,
            "total_lifetime": self.total_lifetime,
            "max_lifetime": self.max_lifetime,
            "intervals": [
                {"birth": iv.birth, "death": iv.death, "lifetime": iv.lifetime}
                for iv in self.intervals
            ],
        }


@dataclass
class TDAResult:
    """
    Result of persistent homology computation.

    Contains persistence diagrams for each computed dimension,
    plus metadata about the computation.

    Attributes:
        diagrams: Dict mapping dimension k to PersistenceDiagram
        num_points: Number of points in the input point cloud
        embedding_dim: Dimension of the embedding space
        max_homology_dim: Maximum homology dimension computed
        backend: Name of the TDA backend used ("ripser", "gudhi", "fallback")
        max_filtration: Maximum filtration value used
    """
    diagrams: Dict[int, PersistenceDiagram] = field(default_factory=dict)
    num_points: int = 0
    embedding_dim: int = 0
    max_homology_dim: int = 1
    backend: str = "fallback"
    max_filtration: float = float('inf')

    def diagram(self, dim: int) -> PersistenceDiagram:
        """Get persistence diagram for dimension k."""
        if dim not in self.diagrams:
            return PersistenceDiagram(dimension=dim, intervals=[])
        return self.diagrams[dim]

    def betti_at_scale(self, epsilon: float) -> Dict[int, int]:
        """
        Compute Betti numbers at a specific scale parameter.

        β_k(ε) = number of intervals [b, d) with b ≤ ε < d

        Args:
            epsilon: Scale parameter

        Returns:
            Dict mapping dimension to Betti number at that scale
        """
        betti: Dict[int, int] = {}
        for dim, diag in self.diagrams.items():
            count = sum(
                1 for iv in diag.intervals
                if iv.birth <= epsilon and (iv.death > epsilon or np.isinf(iv.death))
            )
            betti[dim] = count
        return betti

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "num_points": self.num_points,
            "embedding_dim": self.embedding_dim,
            "max_homology_dim": self.max_homology_dim,
            "backend": self.backend,
            "diagrams": {k: v.to_dict() for k, v in self.diagrams.items()},
        }


def build_metric_complex(
    embeddings: Union[Dict[str, np.ndarray], np.ndarray, List[np.ndarray]],
    max_dim: int = 1,
    max_filtration: Optional[float] = None,
    backend: str = "auto",
) -> TDAResult:
    """
    Build Vietoris-Rips complex and compute persistent homology.

    Per TDA_MIND_SCANNER_SPEC.md Section 3.2:
    1. Construct point cloud from embeddings
    2. Build Vietoris-Rips filtration
    3. Compute persistent homology up to max_dim

    Args:
        embeddings: Either:
            - Dict[str, np.ndarray] mapping state ids to feature vectors
            - np.ndarray of shape (n_points, d)
            - List of feature vectors
        max_dim: Maximum homology dimension to compute (default 1)
        max_filtration: Maximum filtration parameter (default: auto)
        backend: TDA backend to use ("ripser", "gudhi", "auto", "fallback")

    Returns:
        TDAResult containing persistence diagrams
    """
    # Convert embeddings to point cloud array
    if isinstance(embeddings, dict):
        if not embeddings:
            return _empty_result(max_dim)
        points = np.array(list(embeddings.values()), dtype=np.float64)
    elif isinstance(embeddings, list):
        if not embeddings:
            return _empty_result(max_dim)
        points = np.array(embeddings, dtype=np.float64)
    else:
        points = np.asarray(embeddings, dtype=np.float64)

    if points.ndim == 1:
        points = points.reshape(-1, 1)

    if points.shape[0] < 2:
        return _empty_result(max_dim, num_points=points.shape[0])

    # Select backend
    if backend == "auto":
        if HAS_RIPSER:
            backend = "ripser"
        elif HAS_GUDHI:
            backend = "gudhi"
        else:
            backend = "fallback"

    # Compute max filtration if not provided
    if max_filtration is None:
        # Use diameter of point cloud as default
        from scipy.spatial.distance import pdist
        distances = pdist(points)
        max_filtration = np.max(distances) if len(distances) > 0 else 1.0

    # Dispatch to backend
    if backend == "ripser" and HAS_RIPSER:
        return _compute_ripser(points, max_dim, max_filtration)
    elif backend == "gudhi" and HAS_GUDHI:
        return _compute_gudhi(points, max_dim, max_filtration)
    else:
        return _compute_fallback(points, max_dim, max_filtration)


def _empty_result(max_dim: int, num_points: int = 0) -> TDAResult:
    """Create an empty TDAResult for edge cases."""
    diagrams = {k: PersistenceDiagram(dimension=k) for k in range(max_dim + 1)}
    return TDAResult(
        diagrams=diagrams,
        num_points=num_points,
        embedding_dim=0,
        max_homology_dim=max_dim,
        backend="empty",
    )


def _compute_ripser(
    points: np.ndarray,
    max_dim: int,
    max_filtration: float,
) -> TDAResult:
    """Compute persistent homology using Ripser."""
    result = ripser.ripser(
        points,
        maxdim=max_dim,
        thresh=max_filtration,
    )

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


def _compute_gudhi(
    points: np.ndarray,
    max_dim: int,
    max_filtration: float,
) -> TDAResult:
    """Compute persistent homology using GUDHI."""
    rips = gudhi.RipsComplex(points=points, max_edge_length=max_filtration)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dim + 1)
    simplex_tree.compute_persistence()

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


def _compute_fallback(
    points: np.ndarray,
    max_dim: int,
    max_filtration: float,
) -> TDAResult:
    """
    Fallback computation when no TDA library is available.

    Computes H_0 (connected components) via union-find at discrete scales.
    H_1 and higher are approximated as empty (conservative).
    """
    from scipy.spatial.distance import pdist, squareform

    n = points.shape[0]
    if n < 2:
        return _empty_result(max_dim, num_points=n)

    # Compute pairwise distances
    dist_condensed = pdist(points)
    dist_matrix = squareform(dist_condensed)

    # Get sorted unique distances for filtration scales
    unique_dists = np.unique(dist_condensed)
    unique_dists = unique_dists[unique_dists <= max_filtration]

    # Union-find for H_0
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> bool:
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True

    # Track component births and deaths
    # All components born at 0
    birth_times = {i: 0.0 for i in range(n)}
    death_events: List[Tuple[float, int]] = []  # (death_time, component_id)

    # Sort edges by weight
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dist_matrix[i, j], i, j))
    edges.sort()

    # Process edges in order
    for dist, i, j in edges:
        if dist > max_filtration:
            break
        pi, pj = find(i), find(j)
        if pi != pj:
            # Merge: younger component dies
            if birth_times.get(pi, 0) < birth_times.get(pj, 0):
                dying = pj
            else:
                dying = pi
            death_events.append((dist, dying))
            union(i, j)

    # Build H_0 diagram
    h0_intervals = []
    # Essential feature: one component survives
    surviving_roots = set(find(i) for i in range(n))
    for root in surviving_roots:
        h0_intervals.append(PersistenceInterval(
            birth=0.0,
            death=float('inf'),
            dimension=0,
        ))
        break  # Only one essential

    # Finite intervals from merges
    for death_time, _ in death_events:
        h0_intervals.append(PersistenceInterval(
            birth=0.0,
            death=death_time,
            dimension=0,
        ))

    diagrams: Dict[int, PersistenceDiagram] = {
        0: PersistenceDiagram(dimension=0, intervals=h0_intervals),
    }

    # H_1 and higher: empty (conservative)
    for dim in range(1, max_dim + 1):
        diagrams[dim] = PersistenceDiagram(dimension=dim, intervals=[])

    return TDAResult(
        diagrams=diagrams,
        num_points=n,
        embedding_dim=points.shape[1],
        max_homology_dim=max_dim,
        backend="fallback",
        max_filtration=max_filtration,
    )


def bottleneck_distance(
    dgm1: PersistenceDiagram,
    dgm2: PersistenceDiagram,
) -> float:
    """
    Compute bottleneck distance between two persistence diagrams.

    The bottleneck distance is the infimum over all matchings of the maximum
    cost of any matched pair or unmatched point (to diagonal).

    Args:
        dgm1: First persistence diagram
        dgm2: Second persistence diagram

    Returns:
        Bottleneck distance (float)
    """
    if HAS_PERSIM and bottleneck is not None:
        arr1 = dgm1.to_numpy()
        arr2 = dgm2.to_numpy()

        # Filter out infinite deaths for bottleneck computation
        arr1 = arr1[~np.isinf(arr1[:, 1])] if len(arr1) > 0 else arr1
        arr2 = arr2[~np.isinf(arr2[:, 1])] if len(arr2) > 0 else arr2

        if len(arr1) == 0 and len(arr2) == 0:
            return 0.0
        if len(arr1) == 0:
            return float(np.max(arr2[:, 1] - arr2[:, 0]) / 2) if len(arr2) > 0 else 0.0
        if len(arr2) == 0:
            return float(np.max(arr1[:, 1] - arr1[:, 0]) / 2) if len(arr1) > 0 else 0.0

        return float(bottleneck(arr1, arr2))

    # Fallback: approximate bottleneck distance
    return _bottleneck_fallback(dgm1, dgm2)


def _bottleneck_fallback(
    dgm1: PersistenceDiagram,
    dgm2: PersistenceDiagram,
) -> float:
    """
    Approximate bottleneck distance without persim.

    Uses a greedy matching heuristic. Not exact but provides a reasonable bound.
    """
    lifetimes1 = sorted(dgm1.lifetimes(exclude_essential=True), reverse=True)
    lifetimes2 = sorted(dgm2.lifetimes(exclude_essential=True), reverse=True)

    if not lifetimes1 and not lifetimes2:
        return 0.0

    # Pad shorter list with zeros
    max_len = max(len(lifetimes1), len(lifetimes2))
    lifetimes1.extend([0.0] * (max_len - len(lifetimes1)))
    lifetimes2.extend([0.0] * (max_len - len(lifetimes2)))

    # Compute differences
    diffs = [abs(l1 - l2) / 2 for l1, l2 in zip(lifetimes1, lifetimes2)]

    # Bottleneck is max difference
    return max(diffs) if diffs else 0.0
