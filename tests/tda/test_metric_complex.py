"""
Tests for backend/tda/metric_complex.py.

Tests Vietoris-Rips filtration and persistent homology computation.

Per TDA_MIND_SCANNER_SPEC.md Section 3.2:
- Point cloud construction from embeddings
- Vietoris-Rips complex at increasing scales
- Persistence diagram extraction
- Bottleneck distance computation

Determinism requirement: same inputs must produce identical outputs.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestPersistenceInterval:
    """Tests for PersistenceInterval dataclass."""

    def test_lifetime_finite(self) -> None:
        """Finite interval has correct lifetime."""
        from backend.tda.metric_complex import PersistenceInterval

        iv = PersistenceInterval(birth=0.5, death=1.5, dimension=1)
        assert iv.lifetime == 1.0
        assert not iv.is_essential

    def test_lifetime_infinite(self) -> None:
        """Essential feature has infinite lifetime."""
        from backend.tda.metric_complex import PersistenceInterval

        iv = PersistenceInterval(birth=0.0, death=float('inf'), dimension=0)
        assert iv.lifetime == float('inf')
        assert iv.is_essential

    def test_to_tuple(self) -> None:
        """Interval converts to (birth, death) tuple."""
        from backend.tda.metric_complex import PersistenceInterval

        iv = PersistenceInterval(birth=0.1, death=0.9, dimension=1)
        assert iv.to_tuple() == (0.1, 0.9)


class TestPersistenceDiagram:
    """Tests for PersistenceDiagram dataclass."""

    def test_empty_diagram(self) -> None:
        """Empty diagram has zero intervals."""
        from backend.tda.metric_complex import PersistenceDiagram

        diag = PersistenceDiagram(dimension=1)
        assert diag.num_intervals == 0
        assert diag.total_lifetime == 0.0
        assert diag.max_lifetime == 0.0

    def test_diagram_with_intervals(self) -> None:
        """Diagram with intervals computes lifetimes correctly."""
        from backend.tda.metric_complex import PersistenceDiagram, PersistenceInterval

        intervals = [
            PersistenceInterval(birth=0.0, death=1.0, dimension=1),
            PersistenceInterval(birth=0.5, death=2.0, dimension=1),
        ]
        diag = PersistenceDiagram(dimension=1, intervals=intervals)

        assert diag.num_intervals == 2
        assert diag.total_lifetime == 2.5  # 1.0 + 1.5
        assert diag.max_lifetime == 1.5

    def test_lifetimes_exclude_essential(self) -> None:
        """lifetimes() excludes infinite by default."""
        from backend.tda.metric_complex import PersistenceDiagram, PersistenceInterval

        intervals = [
            PersistenceInterval(birth=0.0, death=1.0, dimension=0),
            PersistenceInterval(birth=0.0, death=float('inf'), dimension=0),
        ]
        diag = PersistenceDiagram(dimension=0, intervals=intervals)

        finite = diag.lifetimes(exclude_essential=True)
        assert len(finite) == 1
        assert finite[0] == 1.0

    def test_to_numpy(self) -> None:
        """Diagram converts to numpy array."""
        from backend.tda.metric_complex import PersistenceDiagram, PersistenceInterval

        intervals = [
            PersistenceInterval(birth=0.0, death=1.0, dimension=1),
            PersistenceInterval(birth=0.5, death=1.5, dimension=1),
        ]
        diag = PersistenceDiagram(dimension=1, intervals=intervals)

        arr = diag.to_numpy()
        assert arr.shape == (2, 2)
        assert arr[0, 0] == 0.0
        assert arr[0, 1] == 1.0

    def test_to_dict_serialization(self) -> None:
        """Diagram serializes to dictionary."""
        from backend.tda.metric_complex import PersistenceDiagram, PersistenceInterval

        intervals = [PersistenceInterval(birth=0.0, death=1.0, dimension=1)]
        diag = PersistenceDiagram(dimension=1, intervals=intervals)

        d = diag.to_dict()
        assert d["dimension"] == 1
        assert d["num_intervals"] == 1
        assert "intervals" in d


class TestTDAResult:
    """Tests for TDAResult dataclass."""

    def test_empty_result(self) -> None:
        """Empty result has empty diagrams."""
        from backend.tda.metric_complex import TDAResult

        result = TDAResult()
        assert result.num_points == 0
        assert len(result.diagrams) == 0

    def test_diagram_access(self) -> None:
        """diagram() method returns correct diagram or empty."""
        from backend.tda.metric_complex import TDAResult, PersistenceDiagram

        result = TDAResult(
            diagrams={0: PersistenceDiagram(dimension=0)},
            num_points=10,
        )

        assert result.diagram(0).dimension == 0
        assert result.diagram(1).dimension == 1  # Returns empty for missing

    def test_betti_at_scale(self) -> None:
        """betti_at_scale() computes Betti numbers at given epsilon."""
        from backend.tda.metric_complex import (
            TDAResult,
            PersistenceDiagram,
            PersistenceInterval,
        )

        # One component born at 0, dies at inf
        # One component born at 0, dies at 0.5
        intervals = [
            PersistenceInterval(birth=0.0, death=float('inf'), dimension=0),
            PersistenceInterval(birth=0.0, death=0.5, dimension=0),
        ]
        result = TDAResult(
            diagrams={0: PersistenceDiagram(dimension=0, intervals=intervals)},
        )

        # At epsilon=0.3, both are alive
        betti = result.betti_at_scale(0.3)
        assert betti[0] == 2

        # At epsilon=0.6, only one is alive
        betti = result.betti_at_scale(0.6)
        assert betti[0] == 1


class TestBuildMetricComplex:
    """Tests for build_metric_complex function."""

    def test_empty_embeddings(self) -> None:
        """Empty embeddings produce empty result."""
        from backend.tda.metric_complex import build_metric_complex

        result = build_metric_complex({})
        assert result.num_points == 0

    def test_single_point(self) -> None:
        """Single point produces trivial result."""
        from backend.tda.metric_complex import build_metric_complex

        embeddings = {"a": np.array([0.0, 0.0])}
        result = build_metric_complex(embeddings)

        assert result.num_points == 1

    def test_two_points(self) -> None:
        """Two points produce H_0 with merge."""
        from backend.tda.metric_complex import build_metric_complex

        embeddings = {
            "a": np.array([0.0, 0.0]),
            "b": np.array([1.0, 0.0]),
        }
        result = build_metric_complex(embeddings, max_dim=1)

        assert result.num_points == 2
        # Should have H_0 diagram
        h0 = result.diagram(0)
        assert h0.num_intervals >= 1

    def test_triangle_points(self) -> None:
        """Three points in triangle may produce H_1 features."""
        from backend.tda.metric_complex import build_metric_complex

        embeddings = {
            "a": np.array([0.0, 0.0]),
            "b": np.array([1.0, 0.0]),
            "c": np.array([0.5, 0.866]),  # Equilateral triangle
        }
        result = build_metric_complex(embeddings, max_dim=1)

        assert result.num_points == 3
        assert result.embedding_dim == 2

    def test_dict_input(self) -> None:
        """Dict input is handled correctly."""
        from backend.tda.metric_complex import build_metric_complex

        embeddings = {
            "state_0": np.array([0.0, 0.0, 0.0]),
            "state_1": np.array([1.0, 0.0, 0.0]),
            "state_2": np.array([0.0, 1.0, 0.0]),
        }
        result = build_metric_complex(embeddings)

        assert result.num_points == 3
        assert result.embedding_dim == 3

    def test_array_input(self) -> None:
        """Numpy array input is handled correctly."""
        from backend.tda.metric_complex import build_metric_complex

        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
        ])
        result = build_metric_complex(points)

        assert result.num_points == 3

    def test_list_input(self) -> None:
        """List input is handled correctly."""
        from backend.tda.metric_complex import build_metric_complex

        points = [
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
        ]
        result = build_metric_complex(points)

        assert result.num_points == 2

    def test_determinism(self) -> None:
        """Same input produces identical output."""
        from backend.tda.metric_complex import build_metric_complex

        np.random.seed(42)
        embeddings = {f"s_{i}": np.random.randn(5) for i in range(10)}

        r1 = build_metric_complex(embeddings, max_dim=1)
        r2 = build_metric_complex(embeddings, max_dim=1)

        assert r1.num_points == r2.num_points
        assert r1.backend == r2.backend
        # Diagrams should match
        for dim in range(2):
            d1 = r1.diagram(dim)
            d2 = r2.diagram(dim)
            assert d1.num_intervals == d2.num_intervals

    def test_backend_selection(self) -> None:
        """Backend is selected and reported."""
        from backend.tda.metric_complex import build_metric_complex

        embeddings = {"a": np.array([0.0]), "b": np.array([1.0])}
        result = build_metric_complex(embeddings, backend="fallback")

        assert result.backend == "fallback"


class TestBottleneckDistance:
    """Tests for bottleneck_distance function."""

    def test_identical_diagrams(self) -> None:
        """Identical diagrams have zero distance."""
        from backend.tda.metric_complex import (
            PersistenceDiagram,
            PersistenceInterval,
            bottleneck_distance,
        )

        intervals = [PersistenceInterval(birth=0.0, death=1.0, dimension=1)]
        d1 = PersistenceDiagram(dimension=1, intervals=intervals)
        d2 = PersistenceDiagram(dimension=1, intervals=list(intervals))

        dist = bottleneck_distance(d1, d2)
        assert dist == 0.0

    def test_empty_diagrams(self) -> None:
        """Empty diagrams have zero distance."""
        from backend.tda.metric_complex import PersistenceDiagram, bottleneck_distance

        d1 = PersistenceDiagram(dimension=1)
        d2 = PersistenceDiagram(dimension=1)

        dist = bottleneck_distance(d1, d2)
        assert dist == 0.0

    def test_different_diagrams(self) -> None:
        """Different diagrams have positive distance."""
        from backend.tda.metric_complex import (
            PersistenceDiagram,
            PersistenceInterval,
            bottleneck_distance,
        )

        d1 = PersistenceDiagram(
            dimension=1,
            intervals=[PersistenceInterval(birth=0.0, death=1.0, dimension=1)],
        )
        d2 = PersistenceDiagram(
            dimension=1,
            intervals=[PersistenceInterval(birth=0.0, death=2.0, dimension=1)],
        )

        dist = bottleneck_distance(d1, d2)
        assert dist > 0.0
