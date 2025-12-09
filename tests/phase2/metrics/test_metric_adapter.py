# tests/phase2/metrics/test_metric_adapter.py
"""
Phase II Statistical Test Battery - Metric Adapter Tests

Tests for metric adapter patterns that route to different metric functions.

This module tests:
1. The unified compute_metric routing pattern
2. Schema validation and error handling
3. Type coercion and normalization
4. Edge cases in adapter layer

NO UPLIFT INTERPRETATION: These tests verify mechanical correctness only.
All tests are deterministic and self-contained.
"""

import pytest
from typing import Dict, List, Set, Any, Tuple, Callable

from experiments.slice_success_metrics import (
    compute_goal_hit,
    compute_sparse_success,
    compute_chain_success,
    compute_multi_goal_success,
)

from .conftest import (
    DeterministicGenerator,
    SEED_ADAPTER,
    assert_tuple_bool_float,
    SLICE_PARAMS,
)


# ===========================================================================
# ADAPTER IMPLEMENTATION FOR TESTING
# ===========================================================================

class MetricAdapter:
    """
    Test adapter that routes metric computation based on 'kind' parameter.
    
    This mirrors the production adapter pattern in u2_pipeline.py but is
    self-contained for testing purposes.
    """
    
    VALID_KINDS = {"goal_hit", "density", "chain_length", "multi_goal"}
    
    @classmethod
    def compute_metric(
        cls,
        kind: str,
        verified_hashes: Set[str] = None,
        candidates_tried: int = 0,
        result: Dict[str, Any] = None,
        **kwargs
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Route to appropriate metric function based on kind.
        
        Returns (success, value, details_dict).
        """
        if kind not in cls.VALID_KINDS:
            raise ValueError(f"Unknown metric kind: {kind}")
        
        verified_hashes = verified_hashes or set()
        result = result or {"derivations": []}
        
        details: Dict[str, Any] = {"kind": kind}
        
        if kind == "goal_hit":
            return cls._compute_goal_hit(verified_hashes, result, kwargs, details)
        elif kind == "density":
            return cls._compute_density(verified_hashes, candidates_tried, kwargs, details)
        elif kind == "chain_length":
            return cls._compute_chain_length(verified_hashes, result, kwargs, details)
        elif kind == "multi_goal":
            return cls._compute_multi_goal(verified_hashes, kwargs, details)
        
        raise ValueError(f"Unhandled kind: {kind}")
    
    @classmethod
    def _compute_goal_hit(
        cls,
        verified_hashes: Set[str],
        result: Dict[str, Any],
        kwargs: Dict[str, Any],
        details: Dict[str, Any]
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Compute goal_hit metric."""
        target_hashes = kwargs.get("target_hashes", set())
        min_goal_hits = kwargs.get("min_goal_hits", 1)
        min_total_verified = kwargs.get("min_total_verified", 0)
        
        # Build statements from verified hashes
        statements = [{"hash": h} for h in verified_hashes]
        
        success, value = compute_goal_hit(
            statements,
            target_hashes,
            max(min_goal_hits, min_total_verified)
        )
        
        details["target_count"] = len(target_hashes)
        details["verified_count"] = len(verified_hashes)
        details["hit_count"] = int(value)
        
        return success, value, details
    
    @classmethod
    def _compute_density(
        cls,
        verified_hashes: Set[str],
        candidates_tried: int,
        kwargs: Dict[str, Any],
        details: Dict[str, Any]
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Compute density (sparse_success) metric."""
        min_verified = kwargs.get("min_verified", 1)
        max_candidates = kwargs.get("max_candidates", float('inf'))
        
        success, value = compute_sparse_success(
            len(verified_hashes),
            candidates_tried,
            min_verified
        )
        
        details["verified_count"] = len(verified_hashes)
        details["candidates_tried"] = candidates_tried
        
        # Compute density ratio
        if candidates_tried > 0:
            details["density_ratio"] = len(verified_hashes) / candidates_tried
        else:
            details["density_ratio"] = 0.0
        
        return success, value, details
    
    @classmethod
    def _compute_chain_length(
        cls,
        verified_hashes: Set[str],
        result: Dict[str, Any],
        kwargs: Dict[str, Any],
        details: Dict[str, Any]
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Compute chain_length metric."""
        chain_target_hash = kwargs.get("chain_target_hash", "")
        min_chain_length = kwargs.get("min_chain_length", 1)
        
        # Extract dependency graph from result
        derivations = result.get("derivations", [])
        dep_graph: Dict[str, List[str]] = {}
        for deriv in derivations:
            h = deriv.get("hash", "")
            premises = deriv.get("premises", [])
            if premises:
                dep_graph[h] = premises
        
        statements = [{"hash": h} for h in verified_hashes]
        
        success, value = compute_chain_success(
            statements,
            dep_graph,
            chain_target_hash,
            min_chain_length
        )
        
        details["target_hash"] = chain_target_hash
        details["chain_length"] = int(value)
        details["graph_nodes"] = len(dep_graph)
        
        return success, value, details
    
    @classmethod
    def _compute_multi_goal(
        cls,
        verified_hashes: Set[str],
        kwargs: Dict[str, Any],
        details: Dict[str, Any]
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Compute multi_goal metric."""
        required_goal_hashes = kwargs.get("required_goal_hashes", set())
        
        success, value = compute_multi_goal_success(
            verified_hashes,
            required_goal_hashes
        )
        
        details["required_count"] = len(required_goal_hashes)
        details["met_count"] = int(value)
        
        return success, value, details


# ===========================================================================
# BASIC ADAPTER ROUTING TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
class TestAdapterRouting:
    """Tests for adapter routing to correct metric functions."""

    def test_routes_to_goal_hit(self):
        """Adapter routes 'goal_hit' to compute_goal_hit."""
        verified = {"h1", "h2", "h3"}
        target = {"h1", "h2"}
        
        success, value, details = MetricAdapter.compute_metric(
            kind="goal_hit",
            verified_hashes=verified,
            target_hashes=target,
            min_goal_hits=2
        )
        
        assert success is True
        assert value == 2.0
        assert details["kind"] == "goal_hit"

    def test_routes_to_density(self):
        """Adapter routes 'density' to compute_sparse_success."""
        verified = {"h1", "h2", "h3", "h4", "h5"}
        
        success, value, details = MetricAdapter.compute_metric(
            kind="density",
            verified_hashes=verified,
            candidates_tried=100,
            min_verified=5
        )
        
        assert success is True
        assert value == 5.0
        assert details["kind"] == "density"

    def test_routes_to_chain_length(self):
        """Adapter routes 'chain_length' to compute_chain_success."""
        verified = {"h0", "h1", "h2"}
        result = {
            "derivations": [
                {"hash": "h0", "premises": []},
                {"hash": "h1", "premises": ["h0"]},
                {"hash": "h2", "premises": ["h1"]},
            ]
        }
        
        success, value, details = MetricAdapter.compute_metric(
            kind="chain_length",
            verified_hashes=verified,
            result=result,
            chain_target_hash="h2",
            min_chain_length=3
        )
        
        assert success is True
        assert value == 3.0
        assert details["kind"] == "chain_length"

    def test_routes_to_multi_goal(self):
        """Adapter routes 'multi_goal' to compute_multi_goal_success."""
        verified = {"h1", "h2", "h3"}
        required = {"h1", "h2"}
        
        success, value, details = MetricAdapter.compute_metric(
            kind="multi_goal",
            verified_hashes=verified,
            required_goal_hashes=required
        )
        
        assert success is True
        assert value == 2.0
        assert details["kind"] == "multi_goal"

    def test_invalid_kind_raises(self):
        """Invalid metric kind raises ValueError."""
        with pytest.raises(ValueError, match="Unknown metric kind"):
            MetricAdapter.compute_metric(kind="invalid_kind")


# ===========================================================================
# ADAPTER TYPE COERCION TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.type_stability
class TestAdapterTypeCoercion:
    """Tests for type coercion in adapter layer."""

    def test_return_type_tuple(self):
        """Adapter returns 3-tuple (bool, float, dict)."""
        result = MetricAdapter.compute_metric(
            kind="goal_hit",
            verified_hashes={"h1"},
            target_hashes={"h1"},
            min_goal_hits=1
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)
        assert isinstance(result[2], dict)

    def test_details_dict_has_kind(self):
        """Details dict always contains 'kind' key."""
        for kind in ["goal_hit", "density", "chain_length", "multi_goal"]:
            _, _, details = MetricAdapter.compute_metric(
                kind=kind,
                verified_hashes=set()
            )
            assert "kind" in details
            assert details["kind"] == kind

    def test_value_is_float_not_int(self):
        """Value is always float type."""
        _, value, _ = MetricAdapter.compute_metric(
            kind="goal_hit",
            verified_hashes={"h1", "h2"},
            target_hashes={"h1", "h2"},
            min_goal_hits=2
        )
        assert type(value) is float


# ===========================================================================
# ADAPTER DEFAULT VALUE TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
class TestAdapterDefaults:
    """Tests for adapter default value handling."""

    def test_empty_verified_hashes_default(self):
        """None verified_hashes defaults to empty set."""
        success, value, _ = MetricAdapter.compute_metric(
            kind="goal_hit",
            verified_hashes=None,
            target_hashes={"h1"},
            min_goal_hits=1
        )
        assert success is False
        assert value == 0.0

    def test_empty_result_default(self):
        """None result defaults to empty derivations."""
        success, value, details = MetricAdapter.compute_metric(
            kind="chain_length",
            verified_hashes={"h0"},
            result=None,
            chain_target_hash="h0",
            min_chain_length=1
        )
        assert success is True
        assert value == 1.0
        assert details["graph_nodes"] == 0

    def test_zero_candidates_default(self):
        """Zero candidates_tried is handled."""
        success, value, details = MetricAdapter.compute_metric(
            kind="density",
            verified_hashes={"h1"},
            candidates_tried=0,
            min_verified=0
        )
        assert success is True
        assert value == 1.0
        assert details["density_ratio"] == 0.0


# ===========================================================================
# ADAPTER DETAILS TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
class TestAdapterDetails:
    """Tests for adapter details dict contents."""

    def test_goal_hit_details(self):
        """goal_hit includes expected details."""
        _, _, details = MetricAdapter.compute_metric(
            kind="goal_hit",
            verified_hashes={"h1", "h2", "h3"},
            target_hashes={"h1", "h4"},
            min_goal_hits=1
        )
        
        assert details["kind"] == "goal_hit"
        assert details["target_count"] == 2
        assert details["verified_count"] == 3
        assert details["hit_count"] == 1

    def test_density_details(self):
        """density includes expected details."""
        _, _, details = MetricAdapter.compute_metric(
            kind="density",
            verified_hashes={"h1", "h2"},
            candidates_tried=100,
            min_verified=2
        )
        
        assert details["kind"] == "density"
        assert details["verified_count"] == 2
        assert details["candidates_tried"] == 100
        assert details["density_ratio"] == pytest.approx(0.02)

    def test_chain_length_details(self):
        """chain_length includes expected details."""
        result = {
            "derivations": [
                {"hash": "h0", "premises": []},
                {"hash": "h1", "premises": ["h0"]},
            ]
        }
        _, _, details = MetricAdapter.compute_metric(
            kind="chain_length",
            verified_hashes={"h0", "h1"},
            result=result,
            chain_target_hash="h1",
            min_chain_length=2
        )
        
        assert details["kind"] == "chain_length"
        assert details["target_hash"] == "h1"
        assert details["chain_length"] == 2
        assert details["graph_nodes"] == 1  # Only h1 has premises

    def test_multi_goal_details(self):
        """multi_goal includes expected details."""
        _, _, details = MetricAdapter.compute_metric(
            kind="multi_goal",
            verified_hashes={"h1", "h2", "h3"},
            required_goal_hashes={"h1", "h2", "h4"}
        )
        
        assert details["kind"] == "multi_goal"
        assert details["required_count"] == 3
        assert details["met_count"] == 2


# ===========================================================================
# ADAPTER DETERMINISM TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.determinism
class TestAdapterDeterminism:
    """Determinism tests for adapter layer."""

    def test_same_input_same_output_all_kinds(self):
        """Same inputs produce same outputs for all metric kinds."""
        verified = {"h1", "h2", "h3"}
        
        test_cases = [
            ("goal_hit", {"target_hashes": {"h1", "h2"}, "min_goal_hits": 2}),
            ("density", {"candidates_tried": 100, "min_verified": 3}),
            ("chain_length", {
                "chain_target_hash": "h1",
                "min_chain_length": 1,
                "result": {"derivations": []}
            }),
            ("multi_goal", {"required_goal_hashes": {"h1", "h2"}}),
        ]
        
        for kind, kwargs in test_cases:
            results = [
                MetricAdapter.compute_metric(kind=kind, verified_hashes=verified, **kwargs)
                for _ in range(50)
            ]
            assert all(r == results[0] for r in results), f"Non-deterministic for {kind}"

    def test_deterministic_with_seeded_data(self, gen_adapter: DeterministicGenerator):
        """Deterministic with seeded test data."""
        gen = gen_adapter
        
        # First run
        gen.reset()
        verified1 = gen.hash_set(20)
        targets1 = gen.hash_set(10)
        result1 = MetricAdapter.compute_metric(
            kind="goal_hit",
            verified_hashes=verified1,
            target_hashes=targets1,
            min_goal_hits=5
        )
        
        # Replay
        gen.reset()
        verified2 = gen.hash_set(20)
        targets2 = gen.hash_set(10)
        result2 = MetricAdapter.compute_metric(
            kind="goal_hit",
            verified_hashes=verified2,
            target_hashes=targets2,
            min_goal_hits=5
        )
        
        assert result1 == result2


# ===========================================================================
# ADAPTER CROSS-SLICE TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.cross_slice
class TestAdapterCrossSlice:
    """Cross-slice parameter tests for adapter layer."""

    @pytest.mark.parametrize("slice_id", list(SLICE_PARAMS.keys()))
    def test_adapter_with_slice_params(self, slice_id: str):
        """Test adapter with each slice's parameters."""
        params = SLICE_PARAMS[slice_id]
        min_samples = params["min_samples"]
        
        verified = {f"h{i}" for i in range(min_samples)}
        target = {f"h{i}" for i in range(min_samples // 2)}
        
        success, value, details = MetricAdapter.compute_metric(
            kind="goal_hit",
            verified_hashes=verified,
            target_hashes=target,
            min_goal_hits=min_samples // 2
        )
        
        assert success is True
        assert value == float(min_samples // 2)

    @pytest.mark.parametrize("kind", ["goal_hit", "density", "chain_length", "multi_goal"])
    def test_all_kinds_with_empty_data(self, kind: str):
        """All metric kinds handle empty data."""
        if kind == "density":
            result = MetricAdapter.compute_metric(
                kind=kind,
                verified_hashes=set(),
                candidates_tried=0,
                min_verified=0
            )
        else:
            result = MetricAdapter.compute_metric(
                kind=kind,
                verified_hashes=set()
            )
        
        assert isinstance(result, tuple)
        assert len(result) == 3


# ===========================================================================
# ADAPTER INTEGRATION TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
class TestAdapterIntegration:
    """Integration tests combining multiple adapter calls."""

    def test_sequential_metrics_same_data(self):
        """Compute multiple metrics on same verified set."""
        verified = {"h1", "h2", "h3", "h4", "h5"}
        
        # Goal hit
        gh_success, gh_value, _ = MetricAdapter.compute_metric(
            kind="goal_hit",
            verified_hashes=verified,
            target_hashes={"h1", "h3"},
            min_goal_hits=2
        )
        
        # Density
        d_success, d_value, _ = MetricAdapter.compute_metric(
            kind="density",
            verified_hashes=verified,
            candidates_tried=100,
            min_verified=5
        )
        
        # Multi-goal
        mg_success, mg_value, _ = MetricAdapter.compute_metric(
            kind="multi_goal",
            verified_hashes=verified,
            required_goal_hashes={"h1", "h2"}
        )
        
        assert gh_success is True and gh_value == 2.0
        assert d_success is True and d_value == 5.0
        assert mg_success is True and mg_value == 2.0

    def test_batch_metric_computation(self, gen_adapter: DeterministicGenerator):
        """Batch of metric computations."""
        gen = gen_adapter
        gen.reset()
        
        results = []
        for _ in range(20):
            verified = gen.hash_set(gen.int_value(5, 30))
            targets = gen.hash_set(gen.int_value(2, 10))
            
            result = MetricAdapter.compute_metric(
                kind="goal_hit",
                verified_hashes=verified,
                target_hashes=targets,
                min_goal_hits=1
            )
            results.append(result)
        
        # All results should be valid tuples
        for r in results:
            assert isinstance(r, tuple)
            assert len(r) == 3
            assert isinstance(r[0], bool)
            assert isinstance(r[1], float)
            assert isinstance(r[2], dict)


# ===========================================================================
# ADAPTER ERROR HANDLING TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
class TestAdapterErrorHandling:
    """Error handling tests for adapter layer."""

    def test_unknown_kind_raises_value_error(self):
        """Unknown metric kind raises ValueError."""
        with pytest.raises(ValueError):
            MetricAdapter.compute_metric(kind="unknown")

    def test_empty_string_kind_raises(self):
        """Empty string kind raises ValueError."""
        with pytest.raises(ValueError):
            MetricAdapter.compute_metric(kind="")

    def test_none_kind_raises(self):
        """None kind raises appropriate error."""
        with pytest.raises((ValueError, TypeError)):
            MetricAdapter.compute_metric(kind=None)  # type: ignore


# ===========================================================================
# REPLAY EQUIVALENCE TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.replay
class TestAdapterReplay:
    """Replay equivalence tests for adapter layer."""

    def test_replay_goal_hit_sequence(self, gen_adapter: DeterministicGenerator):
        """Goal hit sequence replays identically."""
        gen = gen_adapter
        
        # First run
        gen.reset()
        sequence1 = []
        for _ in range(30):
            verified = gen.hash_set(20)
            targets = gen.hash_set(5)
            result = MetricAdapter.compute_metric(
                kind="goal_hit",
                verified_hashes=verified,
                target_hashes=targets,
                min_goal_hits=3
            )
            sequence1.append(result)
        
        # Replay
        gen.reset()
        sequence2 = []
        for _ in range(30):
            verified = gen.hash_set(20)
            targets = gen.hash_set(5)
            result = MetricAdapter.compute_metric(
                kind="goal_hit",
                verified_hashes=verified,
                target_hashes=targets,
                min_goal_hits=3
            )
            sequence2.append(result)
        
        assert sequence1 == sequence2

    def test_replay_mixed_kinds(self, gen_adapter: DeterministicGenerator):
        """Mixed metric kinds replay identically."""
        gen = gen_adapter
        kinds = ["goal_hit", "density", "multi_goal"]
        
        # First run
        gen.reset()
        sequence1 = []
        for _ in range(20):
            kind = gen.choice(kinds)
            verified = gen.hash_set(15)
            
            if kind == "goal_hit":
                result = MetricAdapter.compute_metric(
                    kind=kind,
                    verified_hashes=verified,
                    target_hashes=gen.hash_set(5),
                    min_goal_hits=2
                )
            elif kind == "density":
                result = MetricAdapter.compute_metric(
                    kind=kind,
                    verified_hashes=verified,
                    candidates_tried=gen.int_value(50, 200),
                    min_verified=5
                )
            else:  # multi_goal
                result = MetricAdapter.compute_metric(
                    kind=kind,
                    verified_hashes=verified,
                    required_goal_hashes=gen.hash_set(5)
                )
            sequence1.append(result)
        
        # Replay
        gen.reset()
        sequence2 = []
        for _ in range(20):
            kind = gen.choice(kinds)
            verified = gen.hash_set(15)
            
            if kind == "goal_hit":
                result = MetricAdapter.compute_metric(
                    kind=kind,
                    verified_hashes=verified,
                    target_hashes=gen.hash_set(5),
                    min_goal_hits=2
                )
            elif kind == "density":
                result = MetricAdapter.compute_metric(
                    kind=kind,
                    verified_hashes=verified,
                    candidates_tried=gen.int_value(50, 200),
                    min_verified=5
                )
            else:  # multi_goal
                result = MetricAdapter.compute_metric(
                    kind=kind,
                    verified_hashes=verified,
                    required_goal_hashes=gen.hash_set(5)
                )
            sequence2.append(result)
        
        assert sequence1 == sequence2

