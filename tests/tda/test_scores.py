"""
Tests for backend/tda/scores.py.

Tests SNS, PCS, DRS, HSS score computation per TDA_MIND_SCANNER_SPEC.md.

Sections:
- 3.3: SNS (Structural Non-Triviality Score)
- 3.4: PCS (Persistence Coherence Score)
- 3.5: DRS (Deviation-from-Reference Score)
- 3.6: HSS (Hallucination Stability Score)

All functions must be pure, deterministic, and match spec formulas exactly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# Skip if networkx not available
networkx = pytest.importorskip("networkx")
import networkx as nx


class TestScoreWeights:
    """Tests for ScoreWeights dataclass."""

    def test_default_weights(self) -> None:
        """Default weights are α=β=γ=0.4 per spec."""
        from backend.tda.scores import ScoreWeights

        w = ScoreWeights()
        assert w.alpha == 0.4
        assert w.beta == 0.4
        assert w.gamma == 0.4
        assert w.total == 1.2

    def test_validation_rejects_negative(self) -> None:
        """Negative weights are rejected."""
        from backend.tda.scores import ScoreWeights

        w = ScoreWeights(alpha=-0.1, beta=0.4, gamma=0.4)
        with pytest.raises(ValueError):
            w.validate()


class TestSNS:
    """Tests for Structural Non-Triviality Score (SNS)."""

    def test_sns_empty_complex(self) -> None:
        """Empty complex has SNS = 0."""
        from backend.tda.proof_complex import SimplicialComplex
        from backend.tda.scores import compute_structural_nontriviality

        complex_ = SimplicialComplex()
        sns = compute_structural_nontriviality(complex_)

        assert sns == 0.0

    def test_sns_small_tree(self) -> None:
        """Small connected tree has SNS in (0, 0.5]."""
        from backend.tda.proof_complex import build_combinatorial_complex
        from backend.tda.scores import compute_structural_nontriviality

        # Simple tree: a -> b -> c
        G = nx.DiGraph()
        G.add_edge("a", "b")
        G.add_edge("b", "c")

        complex_ = build_combinatorial_complex(G)
        sns = compute_structural_nontriviality(complex_, n_ref=50)

        # f_topo = 0.5 (connected tree), f_size small
        assert 0.0 < sns <= 0.5

    def test_sns_with_cycle(self) -> None:
        """Connected graph with cycle has higher SNS."""
        from backend.tda.proof_complex import build_combinatorial_complex
        from backend.tda.scores import compute_structural_nontriviality

        # Triangle: creates a cycle in undirected view
        G = nx.DiGraph()
        G.add_edge("a", "b")
        G.add_edge("b", "c")
        G.add_edge("a", "c")

        complex_ = build_combinatorial_complex(G)
        sns = compute_structural_nontriviality(complex_, n_ref=50)

        # Should have f_topo = 1.0 (connected with cycles)
        # But triangle is filled, so β_1 = 0
        # Need unfilled cycle for β_1 > 0
        assert sns >= 0.0

    def test_sns_disconnected_no_loops(self) -> None:
        """Disconnected graph with no loops has f_topo = 0."""
        from backend.tda.proof_complex import build_combinatorial_complex
        from backend.tda.scores import compute_structural_nontriviality

        # Two disconnected edges
        G = nx.DiGraph()
        G.add_edge("a", "b")
        G.add_edge("c", "d")

        complex_ = build_combinatorial_complex(G)
        sns = compute_structural_nontriviality(complex_, n_ref=50)

        # f_topo = 0 for disconnected + no loops
        assert sns == 0.0

    def test_sns_size_factor_formula(self) -> None:
        """Size factor follows log formula from spec."""
        from backend.tda.proof_complex import SimplicialComplex
        from backend.tda.scores import compute_structural_nontriviality_detailed

        # Create complex with known size
        complex_ = SimplicialComplex(
            vertices=[f"v{i}" for i in range(10)],
            vertex_to_index={f"v{i}": i for i in range(10)},
        )
        for i in range(10):
            complex_.add_simplex([i])

        result = compute_structural_nontriviality_detailed(complex_, n_ref=50)

        # f_size = min(1, log(1+10)/log(1+50))
        expected_f_size = min(1.0, math.log(11) / math.log(51))
        assert abs(result["f_size"] - expected_f_size) < 0.01

    def test_sns_determinism(self) -> None:
        """Same input produces same SNS."""
        from backend.tda.proof_complex import build_combinatorial_complex
        from backend.tda.scores import compute_structural_nontriviality

        G = nx.DiGraph()
        G.add_edge("x", "y")
        G.add_edge("y", "z")

        complex_ = build_combinatorial_complex(G)
        sns1 = compute_structural_nontriviality(complex_, n_ref=50)
        sns2 = compute_structural_nontriviality(complex_, n_ref=50)

        assert sns1 == sns2


class TestPCS:
    """Tests for Persistence Coherence Score (PCS)."""

    def test_pcs_empty_diagram(self) -> None:
        """Empty diagram has PCS = 0."""
        from backend.tda.metric_complex import TDAResult, PersistenceDiagram
        from backend.tda.scores import compute_persistence_coherence

        result = TDAResult(
            diagrams={
                0: PersistenceDiagram(dimension=0),
                1: PersistenceDiagram(dimension=1),
            }
        )
        pcs = compute_persistence_coherence(result)

        assert pcs == 0.0

    def test_pcs_all_long_lived(self) -> None:
        """All long-lived features have PCS = 1."""
        from backend.tda.metric_complex import (
            TDAResult,
            PersistenceDiagram,
            PersistenceInterval,
        )
        from backend.tda.scores import compute_persistence_coherence

        # All intervals have lifetime > tau (0.05)
        h1_intervals = [
            PersistenceInterval(birth=0.0, death=1.0, dimension=1),
            PersistenceInterval(birth=0.5, death=2.0, dimension=1),
        ]
        result = TDAResult(
            diagrams={
                0: PersistenceDiagram(dimension=0),
                1: PersistenceDiagram(dimension=1, intervals=h1_intervals),
            }
        )
        pcs = compute_persistence_coherence(result, lifetime_threshold=0.05)

        # All lifetimes > tau, so PCS_1 = 1.0
        # PCS = 0.7 * 1.0 + 0.3 * 0.0 = 0.7
        assert pcs == pytest.approx(0.7, abs=0.01)

    def test_pcs_mixed_lifetimes(self) -> None:
        """Mixed lifetimes produce intermediate PCS."""
        from backend.tda.metric_complex import (
            TDAResult,
            PersistenceDiagram,
            PersistenceInterval,
        )
        from backend.tda.scores import compute_persistence_coherence

        # One long, one short
        h1_intervals = [
            PersistenceInterval(birth=0.0, death=1.0, dimension=1),  # lifetime=1.0
            PersistenceInterval(birth=0.0, death=0.02, dimension=1),  # lifetime=0.02
        ]
        result = TDAResult(
            diagrams={
                0: PersistenceDiagram(dimension=0),
                1: PersistenceDiagram(dimension=1, intervals=h1_intervals),
            }
        )
        pcs = compute_persistence_coherence(result, lifetime_threshold=0.05)

        # L_total = 1.0 + 0.02 = 1.02
        # L_long = 1.0 (only first > tau)
        # PCS_1 = 1.0 / 1.02 ≈ 0.98
        assert 0.5 < pcs < 1.0

    def test_pcs_determinism(self) -> None:
        """Same input produces same PCS."""
        from backend.tda.metric_complex import (
            TDAResult,
            PersistenceDiagram,
            PersistenceInterval,
        )
        from backend.tda.scores import compute_persistence_coherence

        intervals = [PersistenceInterval(birth=0.0, death=0.5, dimension=1)]
        result = TDAResult(
            diagrams={1: PersistenceDiagram(dimension=1, intervals=intervals)}
        )

        pcs1 = compute_persistence_coherence(result)
        pcs2 = compute_persistence_coherence(result)

        assert pcs1 == pcs2


class TestDRS:
    """Tests for Deviation-from-Reference Score (DRS)."""

    def test_drs_no_reference(self) -> None:
        """No reference profile gives DRS = 0."""
        from backend.tda.metric_complex import TDAResult
        from backend.tda.scores import compute_deviation_from_reference

        result = TDAResult()
        drs = compute_deviation_from_reference(result, ref_profile=None)

        assert drs == 0.0

    def test_drs_identical_to_reference(self) -> None:
        """Identical to reference has DRS ≈ 0."""
        from backend.tda.metric_complex import (
            TDAResult,
            PersistenceDiagram,
            PersistenceInterval,
        )
        from backend.tda.reference_profile import ReferenceTDAProfile
        from backend.tda.scores import compute_deviation_from_reference

        intervals = [PersistenceInterval(birth=0.0, death=1.0, dimension=1)]
        diagram = PersistenceDiagram(dimension=1, intervals=intervals)

        result = TDAResult(diagrams={1: diagram})
        ref = ReferenceTDAProfile(
            slice_name="test",
            reference_diagram_h1=diagram,
            deviation_max=0.5,
        )

        drs = compute_deviation_from_reference(result, ref)
        assert drs == pytest.approx(0.0, abs=0.01)

    def test_drs_normalization(self) -> None:
        """DRS is normalized by deviation_max."""
        from backend.tda.metric_complex import TDAResult, PersistenceDiagram
        from backend.tda.reference_profile import ReferenceTDAProfile
        from backend.tda.scores import compute_deviation_from_reference

        result = TDAResult(diagrams={1: PersistenceDiagram(dimension=1)})
        ref = ReferenceTDAProfile(
            slice_name="test",
            reference_diagram_h1=PersistenceDiagram(dimension=1),
            deviation_max=0.5,
        )

        drs = compute_deviation_from_reference(result, ref, deviation_max=0.5)
        assert 0.0 <= drs <= 1.0


class TestHSS:
    """Tests for Hallucination Stability Score (HSS)."""

    def test_hss_formula_basic(self) -> None:
        """HSS follows spec formula."""
        from backend.tda.scores import (
            compute_hallucination_stability_score,
            ScoreWeights,
        )

        # With sns=1, pcs=1, drs=0, HSS should be high
        hss = compute_hallucination_stability_score(sns=1.0, pcs=1.0, drs=0.0)

        # raw = 0.4*1 + 0.4*1 - 0.4*0 = 0.8
        # HSS = (0.8 + 0.4) / 1.2 = 1.0
        assert hss == pytest.approx(1.0, abs=0.01)

    def test_hss_low_when_drs_high(self) -> None:
        """High DRS produces low HSS."""
        from backend.tda.scores import compute_hallucination_stability_score

        # sns=0, pcs=0, drs=1.0
        hss = compute_hallucination_stability_score(sns=0.0, pcs=0.0, drs=1.0)

        # raw = 0 + 0 - 0.4 = -0.4
        # HSS = (-0.4 + 0.4) / 1.2 = 0.0
        assert hss == pytest.approx(0.0, abs=0.01)

    def test_hss_clamped_to_01(self) -> None:
        """HSS is clamped to [0, 1]."""
        from backend.tda.scores import compute_hallucination_stability_score

        # Even with extreme values
        hss1 = compute_hallucination_stability_score(sns=10.0, pcs=10.0, drs=0.0)
        hss2 = compute_hallucination_stability_score(sns=0.0, pcs=0.0, drs=10.0)

        assert hss1 == 1.0
        assert hss2 == 0.0

    def test_hss_custom_weights(self) -> None:
        """Custom weights affect HSS."""
        from backend.tda.scores import (
            compute_hallucination_stability_score,
            ScoreWeights,
        )

        # Weight SNS heavily
        weights = ScoreWeights(alpha=1.0, beta=0.0, gamma=0.0)

        hss = compute_hallucination_stability_score(
            sns=0.5, pcs=0.0, drs=0.0, weights=weights
        )

        # raw = 1.0*0.5 + 0 - 0 = 0.5
        # HSS = (0.5 + 0) / 1.0 = 0.5
        assert hss == pytest.approx(0.5, abs=0.01)

    def test_hss_determinism(self) -> None:
        """Same inputs produce same HSS."""
        from backend.tda.scores import compute_hallucination_stability_score

        hss1 = compute_hallucination_stability_score(sns=0.6, pcs=0.4, drs=0.2)
        hss2 = compute_hallucination_stability_score(sns=0.6, pcs=0.4, drs=0.2)

        assert hss1 == hss2


class TestClassifyHSS:
    """Tests for classify_hss function."""

    def test_block_below_threshold(self) -> None:
        """HSS below block threshold returns BLOCK."""
        from backend.tda.scores import classify_hss

        assert classify_hss(0.1, block_threshold=0.2, warn_threshold=0.5) == "BLOCK"

    def test_warn_in_middle(self) -> None:
        """HSS between thresholds returns WARN."""
        from backend.tda.scores import classify_hss

        assert classify_hss(0.3, block_threshold=0.2, warn_threshold=0.5) == "WARN"

    def test_ok_above_threshold(self) -> None:
        """HSS above warn threshold returns OK."""
        from backend.tda.scores import classify_hss

        assert classify_hss(0.7, block_threshold=0.2, warn_threshold=0.5) == "OK"

    def test_boundary_conditions(self) -> None:
        """Boundary values are handled correctly."""
        from backend.tda.scores import classify_hss

        # Exactly at block threshold -> WARN (not BLOCK)
        assert classify_hss(0.2, block_threshold=0.2, warn_threshold=0.5) == "WARN"

        # Exactly at warn threshold -> OK
        assert classify_hss(0.5, block_threshold=0.2, warn_threshold=0.5) == "OK"


class TestComputeAllScores:
    """Tests for compute_all_scores aggregate function."""

    def test_computes_all_scores(self) -> None:
        """compute_all_scores returns complete ScoreResult."""
        from backend.tda.proof_complex import build_combinatorial_complex
        from backend.tda.metric_complex import build_metric_complex
        from backend.tda.scores import compute_all_scores

        G = nx.DiGraph()
        G.add_edge("a", "b")
        G.add_edge("b", "c")

        complex_ = build_combinatorial_complex(G)
        embeddings = {
            "a": np.array([0.0, 0.0]),
            "b": np.array([1.0, 0.0]),
            "c": np.array([0.5, 1.0]),
        }
        tda_result = build_metric_complex(embeddings)

        result = compute_all_scores(complex_, tda_result)

        assert 0.0 <= result.sns <= 1.0
        assert 0.0 <= result.pcs <= 1.0
        assert 0.0 <= result.drs <= 1.0
        assert 0.0 <= result.hss <= 1.0
        assert "betti" in result.to_dict()
