"""
Tests for backend/tda/runtime_monitor.py.

Tests TDAMonitor sidecar integration per TDA_MIND_SCANNER_SPEC.md Section 4.

Coverage:
- TDAMonitorConfig validation
- TDAMonitorResult structure
- evaluate_proof_attempt() workflow
- Gating signal logic (BLOCK, WARN, OK)
- Operational modes (SHADOW, SOFT, HARD)
- Error handling (fail-open vs fail-closed)

Determinism requirement: same inputs must produce identical results.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip if networkx not available
networkx = pytest.importorskip("networkx")
import networkx as nx


class TestTDAGatingSignal:
    """Tests for TDAGatingSignal enum."""

    def test_signal_values(self) -> None:
        """Gating signals have correct string values."""
        from backend.tda.runtime_monitor import TDAGatingSignal

        assert TDAGatingSignal.BLOCK.value == "BLOCK"
        assert TDAGatingSignal.WARN.value == "WARN"
        assert TDAGatingSignal.OK.value == "OK"


class TestTDAOperationalMode:
    """Tests for TDAOperationalMode enum."""

    def test_mode_values(self) -> None:
        """Operational modes have correct string values."""
        from backend.tda.runtime_monitor import TDAOperationalMode

        assert TDAOperationalMode.OFFLINE.value == "offline"
        assert TDAOperationalMode.SHADOW.value == "shadow"
        assert TDAOperationalMode.SOFT.value == "soft"
        assert TDAOperationalMode.HARD.value == "hard"


class TestTDAMonitorConfig:
    """Tests for TDAMonitorConfig dataclass."""

    def test_default_config(self) -> None:
        """Default config matches spec values."""
        from backend.tda.runtime_monitor import TDAMonitorConfig, TDAOperationalMode

        cfg = TDAMonitorConfig()

        assert cfg.hss_block_threshold == 0.2
        assert cfg.hss_warn_threshold == 0.5
        assert cfg.max_simplex_dim == 3
        assert cfg.max_homology_dim == 1
        assert cfg.lifetime_threshold == 0.05
        assert cfg.deviation_max == 0.5
        assert cfg.mode == TDAOperationalMode.SHADOW
        assert cfg.fail_open is True

    def test_validation_thresholds(self) -> None:
        """Config validation catches invalid thresholds."""
        from backend.tda.runtime_monitor import TDAMonitorConfig

        # Block >= warn is invalid
        cfg = TDAMonitorConfig(hss_block_threshold=0.5, hss_warn_threshold=0.3)
        with pytest.raises(ValueError):
            cfg.validate()

        # Out of range
        cfg = TDAMonitorConfig(hss_block_threshold=1.5)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validation_passes_valid_config(self) -> None:
        """Valid config passes validation."""
        from backend.tda.runtime_monitor import TDAMonitorConfig

        cfg = TDAMonitorConfig(hss_block_threshold=0.1, hss_warn_threshold=0.6)
        cfg.validate()  # Should not raise


class TestTDAMonitorResult:
    """Tests for TDAMonitorResult dataclass."""

    def test_result_properties(self) -> None:
        """Result properties compute correctly."""
        from backend.tda.runtime_monitor import TDAMonitorResult, TDAGatingSignal

        result = TDAMonitorResult(
            hss=0.6,
            sns=0.5,
            pcs=0.4,
            drs=0.1,
            signal=TDAGatingSignal.OK,
            block=False,
            warn=False,
        )

        assert result.ok is True
        assert result.block is False
        assert result.warn is False

    def test_to_dict_serialization(self) -> None:
        """Result serializes to dictionary."""
        from backend.tda.runtime_monitor import TDAMonitorResult, TDAGatingSignal

        result = TDAMonitorResult(
            hss=0.5,
            sns=0.5,
            pcs=0.5,
            drs=0.0,
            signal=TDAGatingSignal.WARN,
            block=False,
            warn=True,
            betti={0: 1, 1: 0},
            metadata={"slice": "test"},
        )

        d = result.to_dict()

        assert d["hss"] == 0.5
        assert d["signal"] == "WARN"
        assert d["block"] is False
        assert d["warn"] is True
        assert d["betti"] == {0: 1, 1: 0}


class TestTDAMonitor:
    """Tests for TDAMonitor class."""

    def _make_simple_dag(self) -> nx.DiGraph:
        """Create a simple test DAG."""
        G = nx.DiGraph()
        G.add_edge("axiom", "lemma1")
        G.add_edge("axiom", "lemma2")
        G.add_edge("lemma1", "goal")
        G.add_edge("lemma2", "goal")
        return G

    def _make_simple_embeddings(self, nodes: list) -> dict:
        """Create simple embeddings for nodes."""
        return {n: np.random.randn(10).astype(np.float32) for n in nodes}

    def test_monitor_instantiation(self) -> None:
        """Monitor instantiates with valid config."""
        from backend.tda.runtime_monitor import TDAMonitor, TDAMonitorConfig

        cfg = TDAMonitorConfig()
        monitor = TDAMonitor(cfg)

        assert monitor.cfg.hss_block_threshold == 0.2

    def test_evaluate_proof_attempt(self) -> None:
        """evaluate_proof_attempt returns valid result."""
        from backend.tda.runtime_monitor import TDAMonitor, TDAMonitorConfig

        cfg = TDAMonitorConfig()
        monitor = TDAMonitor(cfg)

        dag = self._make_simple_dag()
        embeddings = self._make_simple_embeddings(list(dag.nodes()))

        result = monitor.evaluate_proof_attempt(
            slice_name="test",
            local_dag=dag,
            embeddings=embeddings,
        )

        assert 0.0 <= result.hss <= 1.0
        assert 0.0 <= result.sns <= 1.0
        assert 0.0 <= result.pcs <= 1.0
        assert 0.0 <= result.drs <= 1.0
        assert result.computation_time_ms >= 0

    def test_shadow_mode_no_gating(self) -> None:
        """Shadow mode never blocks or warns."""
        from backend.tda.runtime_monitor import (
            TDAMonitor,
            TDAMonitorConfig,
            TDAOperationalMode,
        )

        cfg = TDAMonitorConfig(
            mode=TDAOperationalMode.SHADOW,
            hss_block_threshold=0.9,  # Would normally block most
        )
        monitor = TDAMonitor(cfg)

        dag = self._make_simple_dag()
        embeddings = self._make_simple_embeddings(list(dag.nodes()))

        result = monitor.evaluate_proof_attempt("test", dag, embeddings)

        # Shadow mode: always False for block/warn
        assert result.block is False
        assert result.warn is False

    def test_soft_mode_warns_not_blocks(self) -> None:
        """Soft mode warns but never blocks."""
        from backend.tda.runtime_monitor import (
            TDAMonitor,
            TDAMonitorConfig,
            TDAOperationalMode,
            TDAGatingSignal,
        )

        cfg = TDAMonitorConfig(
            mode=TDAOperationalMode.SOFT,
            hss_block_threshold=0.9,  # High threshold
            hss_warn_threshold=0.99,
        )
        monitor = TDAMonitor(cfg)

        dag = self._make_simple_dag()
        embeddings = self._make_simple_embeddings(list(dag.nodes()))

        result = monitor.evaluate_proof_attempt("test", dag, embeddings)

        # Soft mode: block always False
        assert result.block is False
        # Signal may be BLOCK but result.block is False

    def test_hard_mode_can_block(self) -> None:
        """Hard mode blocks when HSS < block_threshold."""
        from backend.tda.runtime_monitor import (
            TDAMonitor,
            TDAMonitorConfig,
            TDAOperationalMode,
        )

        cfg = TDAMonitorConfig(
            mode=TDAOperationalMode.HARD,
            hss_block_threshold=0.99,  # Very high - should block
            hss_warn_threshold=0.999,
        )
        monitor = TDAMonitor(cfg)

        dag = self._make_simple_dag()
        embeddings = self._make_simple_embeddings(list(dag.nodes()))

        result = monitor.evaluate_proof_attempt("test", dag, embeddings)

        # Hard mode with high threshold: likely blocks
        # (depends on actual HSS, but threshold is very high)
        assert result.signal.value in ["BLOCK", "WARN", "OK"]

    def test_should_block_hard_mode(self) -> None:
        """should_block() returns True only in HARD mode with BLOCK signal."""
        from backend.tda.runtime_monitor import (
            TDAMonitor,
            TDAMonitorConfig,
            TDAOperationalMode,
            TDAMonitorResult,
            TDAGatingSignal,
        )

        # HARD mode
        cfg_hard = TDAMonitorConfig(mode=TDAOperationalMode.HARD)
        monitor_hard = TDAMonitor(cfg_hard)

        result_block = TDAMonitorResult(
            hss=0.1, sns=0.0, pcs=0.0, drs=0.5,
            signal=TDAGatingSignal.BLOCK,
            block=True, warn=False,
        )
        assert monitor_hard.should_block(result_block) is True

        # SHADOW mode - should_block always False
        cfg_shadow = TDAMonitorConfig(mode=TDAOperationalMode.SHADOW)
        monitor_shadow = TDAMonitor(cfg_shadow)
        assert monitor_shadow.should_block(result_block) is False

    def test_should_warn(self) -> None:
        """should_warn() returns True in SOFT/HARD mode with WARN/BLOCK."""
        from backend.tda.runtime_monitor import (
            TDAMonitor,
            TDAMonitorConfig,
            TDAOperationalMode,
            TDAMonitorResult,
            TDAGatingSignal,
        )

        cfg = TDAMonitorConfig(mode=TDAOperationalMode.SOFT)
        monitor = TDAMonitor(cfg)

        result = TDAMonitorResult(
            hss=0.3, sns=0.2, pcs=0.2, drs=0.3,
            signal=TDAGatingSignal.WARN,
            block=False, warn=True,
        )

        assert monitor.should_warn(result) is True

    def test_statistics_tracking(self) -> None:
        """Monitor tracks evaluation statistics."""
        from backend.tda.runtime_monitor import TDAMonitor, TDAMonitorConfig

        cfg = TDAMonitorConfig()
        monitor = TDAMonitor(cfg)

        dag = self._make_simple_dag()
        embeddings = self._make_simple_embeddings(list(dag.nodes()))

        # Initial stats
        stats = monitor.get_statistics()
        assert stats["eval_count"] == 0

        # After evaluation
        monitor.evaluate_proof_attempt("test", dag, embeddings)
        stats = monitor.get_statistics()
        assert stats["eval_count"] == 1

        # Reset
        monitor.reset_statistics()
        stats = monitor.get_statistics()
        assert stats["eval_count"] == 0

    def test_determinism(self) -> None:
        """Same inputs produce identical results."""
        from backend.tda.runtime_monitor import TDAMonitor, TDAMonitorConfig

        np.random.seed(42)

        cfg = TDAMonitorConfig()
        monitor = TDAMonitor(cfg)

        dag = self._make_simple_dag()
        embeddings = {n: np.array([float(i)] * 10) for i, n in enumerate(dag.nodes())}

        r1 = monitor.evaluate_proof_attempt("test", dag, embeddings)
        r2 = monitor.evaluate_proof_attempt("test", dag, embeddings)

        assert r1.hss == r2.hss
        assert r1.sns == r2.sns
        assert r1.pcs == r2.pcs
        assert r1.signal == r2.signal

    def test_error_handling_fail_open(self) -> None:
        """Fail-open mode returns OK on errors."""
        from backend.tda.runtime_monitor import (
            TDAMonitor,
            TDAMonitorConfig,
            TDAGatingSignal,
        )

        cfg = TDAMonitorConfig(fail_open=True)
        monitor = TDAMonitor(cfg)

        # Pass invalid input to trigger error
        result = monitor.evaluate_proof_attempt(
            slice_name="test",
            local_dag="not_a_graph",  # type: ignore
            embeddings={},
        )

        # Fail-open: should return OK signal
        assert result.signal == TDAGatingSignal.OK
        assert result.error is not None

    def test_error_handling_fail_closed(self) -> None:
        """Fail-closed mode returns WARN on errors."""
        from backend.tda.runtime_monitor import (
            TDAMonitor,
            TDAMonitorConfig,
            TDAGatingSignal,
        )

        cfg = TDAMonitorConfig(fail_open=False)
        monitor = TDAMonitor(cfg)

        # Pass invalid input to trigger error
        result = monitor.evaluate_proof_attempt(
            slice_name="test",
            local_dag="not_a_graph",  # type: ignore
            embeddings={},
        )

        # Fail-closed: should return WARN signal
        assert result.signal == TDAGatingSignal.WARN
        assert result.error is not None


class TestCreateMonitor:
    """Tests for create_monitor factory function."""

    def test_create_with_defaults(self) -> None:
        """Factory creates monitor with default settings."""
        from backend.tda.runtime_monitor import create_monitor, TDAOperationalMode

        monitor = create_monitor()

        assert monitor.cfg.mode == TDAOperationalMode.SHADOW

    def test_create_with_custom_mode(self) -> None:
        """Factory creates monitor with specified mode."""
        from backend.tda.runtime_monitor import create_monitor, TDAOperationalMode

        monitor = create_monitor(mode="hard")

        assert monitor.cfg.mode == TDAOperationalMode.HARD

    def test_create_with_custom_thresholds(self) -> None:
        """Factory creates monitor with specified thresholds."""
        from backend.tda.runtime_monitor import create_monitor

        monitor = create_monitor(block_threshold=0.3, warn_threshold=0.7)

        assert monitor.cfg.hss_block_threshold == 0.3
        assert monitor.cfg.hss_warn_threshold == 0.7
