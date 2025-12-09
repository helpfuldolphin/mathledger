"""
PHASE II — NOT RUN IN PHASE I

Metric Conformance Suite Implementation
=======================================

This module implements the conformance tests specified in:
    docs/METRIC_CONFORMANCE_SUITE_SPEC.md

Test IDs follow the naming convention:
    {METRIC}-T{N}-{INVARIANT}-{PROPERTY}

Conformance Levels:
    L0 (minimal):    Type + determinism tests only (smoke)
    L1 (standard):   All invariant tests
    L2 (full):       Standard + monotonicity + boundary exhaustive
    L3 (exhaustive): Full + stress tests (1000+ iterations)

Usage:
    pytest tests/metrics/test_metric_conformance.py -m "conformance_l0"
    pytest tests/metrics/test_metric_conformance.py -m "conformance_l1"
    pytest tests/metrics/test_metric_conformance.py -m "conformance_l2"
    pytest tests/metrics/test_metric_conformance.py -m "conformance_l3"
    pytest tests/metrics/test_metric_conformance.py  # runs L1 by default
"""

import pytest
from typing import List, Dict, Set, Any, Tuple
from dataclasses import dataclass


# =============================================================================
# WRAPPER FUNCTIONS FOR CONFORMANCE TESTING
# =============================================================================
# These wrappers implement metric computation directly for conformance testing.
# They mirror the logic in backend.substrate.slice_success_metrics but are
# self-contained to avoid import dependencies.

def compute_goal_hit(
    verified_statements: List[Dict[str, Any]],
    target_hashes: Set[str],
    min_total_verified: int = 1,
) -> Tuple[bool, float]:
    """
    Self-contained goal_hit metric implementation for conformance testing.

    Computes success based on hitting a minimum number of specific target goals.
    """
    verified_hashes = {s["hash"] for s in verified_statements}
    hits = len(verified_hashes.intersection(target_hashes))
    success = hits >= min_total_verified
    return success, float(hits)


def compute_sparse_success(
    verified_count: int,
    attempted_count: int,
    min_verified: int = 1,
) -> Tuple[bool, float]:
    """
    Self-contained sparse_success metric implementation for conformance testing.

    Computes success based on a simple minimum count of verified statements.
    """
    _ = attempted_count  # API compatibility
    success = verified_count >= min_verified
    return success, float(verified_count)


def compute_chain_success(
    verified_statements: List[Dict[str, Any]],
    dependency_graph: Dict[str, List[str]],
    target_hash: str,
    min_chain_length: int = 1,
) -> Tuple[bool, float]:
    """
    Wrapper for chain_success metric conformance testing.

    Note: The actual compute_metric("chain_length") requires a different
    interface. This wrapper implements the chain logic directly for
    conformance testing.
    """
    verified_hashes = {s["hash"] for s in verified_statements}

    # If target not verified, chain length is 0
    if target_hash not in verified_hashes:
        return False, 0.0

    # Compute longest chain to target using memoization
    memo: Dict[str, int] = {}
    visiting: Set[str] = set()

    def chain_length(h: str) -> int:
        if h not in verified_hashes:
            return 0
        if h in memo:
            return memo[h]
        if h in visiting:
            # Cycle detected, return 1 (just this node)
            return 1

        visiting.add(h)
        deps = dependency_graph.get(h, [])
        if not deps:
            length = 1
        else:
            max_dep_length = max((chain_length(d) for d in deps), default=0)
            length = 1 + max_dep_length
        visiting.discard(h)
        memo[h] = length
        return length

    length = chain_length(target_hash)
    success = length >= min_chain_length
    return success, float(length)


def compute_multi_goal_success(
    verified_hashes: Set[str],
    required_goal_hashes: Set[str],
) -> Tuple[bool, float]:
    """
    Self-contained multi_goal metric implementation for conformance testing.

    Computes success based on verifying a set of required goals.
    """
    if not required_goal_hashes:
        return True, 0.0

    met_goals = verified_hashes.intersection(required_goal_hashes)
    num_met = len(met_goals)
    success = num_met == len(required_goal_hashes)
    return success, float(num_met)


# =============================================================================
# TEST FACTORY HELPERS
# =============================================================================

@dataclass
class StatementFactory:
    """Factory for generating verified statement test data."""

    @staticmethod
    def make_statements(hashes: List[str]) -> List[Dict[str, Any]]:
        """Create statement list from hash strings."""
        return [{"hash": h} for h in hashes]

    @staticmethod
    def make_range(prefix: str, count: int) -> List[Dict[str, Any]]:
        """Create statements with sequential hashes: {prefix}0, {prefix}1, ..."""
        return [{"hash": f"{prefix}{i}"} for i in range(count)]

    @staticmethod
    def make_targets(prefix: str, count: int) -> Set[str]:
        """Create target hash set: {prefix}0, {prefix}1, ..."""
        return {f"{prefix}{i}" for i in range(count)}


@dataclass
class GraphFactory:
    """Factory for generating dependency graph test data."""

    @staticmethod
    def make_linear_chain(length: int, prefix: str = "h") -> Dict[str, List[str]]:
        """
        Create linear chain: h0 <- h1 <- h2 <- ... <- h{n-1}
        Returns graph where h{i} depends on h{i-1}.
        """
        return {f"{prefix}{i}": [f"{prefix}{i-1}"] for i in range(1, length)}

    @staticmethod
    def make_diamond(prefix: str = "h") -> Dict[str, List[str]]:
        """
        Create diamond graph:
            h0
           / \
          h1  h2
           \ /
            h3
        h3 depends on h1 and h2, both depend on h0.
        """
        return {
            f"{prefix}3": [f"{prefix}1", f"{prefix}2"],
            f"{prefix}1": [f"{prefix}0"],
            f"{prefix}2": [f"{prefix}0"],
        }

    @staticmethod
    def make_cycle(nodes: List[str]) -> Dict[str, List[str]]:
        """Create cyclic graph: n0 -> n1 -> n2 -> ... -> n0"""
        if len(nodes) < 2:
            return {}
        graph = {}
        for i, node in enumerate(nodes):
            next_node = nodes[(i + 1) % len(nodes)]
            graph[node] = [next_node]
        return graph

    @staticmethod
    def make_wide(root: str, children: List[str]) -> Dict[str, List[str]]:
        """Create wide tree: root depends on all children."""
        return {root: children}


# =============================================================================
# PYTEST MARKERS FOR CONFORMANCE LEVELS
# =============================================================================

# Marker shortcuts
l0 = pytest.mark.conformance_l0
l1 = pytest.mark.conformance_l1
l2 = pytest.mark.conformance_l2
l3 = pytest.mark.conformance_l3

# Invariant markers
determinism = pytest.mark.determinism
type_stability = pytest.mark.type_stability
boundary = pytest.mark.boundary


# =============================================================================
# GOAL_HIT CONFORMANCE TESTS (GOAL-T1 through GOAL-T15)
# =============================================================================

class TestGoalHitConformance:
    """
    Conformance tests for compute_goal_hit.
    Reference: METRIC_CONFORMANCE_SUITE_SPEC.md Section 3.
    """

    # -------------------------------------------------------------------------
    # GOAL-T1: Hit count bounded by target count (GOAL-1)
    # -------------------------------------------------------------------------
    @l1
    def test_GOAL_T1_hit_bound(self):
        """GOAL-T1: value <= len(target_hashes) [GOAL-1]"""
        targets = {"t0", "t1", "t2"}  # 3 targets
        statements = StatementFactory.make_range("h", 100)
        # Add some targets to statements
        statements.extend(StatementFactory.make_statements(["t0", "t1", "t2", "t3", "t4"]))

        _, value = compute_goal_hit(statements, targets, min_total_verified=0)

        assert value <= len(targets), f"GOAL-1 violated: value {value} > target count {len(targets)}"

    # -------------------------------------------------------------------------
    # GOAL-T2: Hit count bounded by verified count (GOAL-2)
    # -------------------------------------------------------------------------
    @l1
    def test_GOAL_T2_verified_bound(self):
        """GOAL-T2: value <= len(verified_statements) [GOAL-2]"""
        targets = StatementFactory.make_targets("t", 100)  # Many targets
        statements = StatementFactory.make_statements(["t0", "t1"])  # Few verified

        _, value = compute_goal_hit(statements, targets, min_total_verified=0)

        assert value <= len(statements), f"GOAL-2 violated: value {value} > verified count {len(statements)}"

    # -------------------------------------------------------------------------
    # GOAL-T3: Empty targets yield zero hits (GOAL-3)
    # -------------------------------------------------------------------------
    @l1
    def test_GOAL_T3_empty_targets(self):
        """GOAL-T3: target_hashes={} -> value=0.0 [GOAL-3]"""
        statements = StatementFactory.make_range("h", 50)

        _, value = compute_goal_hit(statements, set(), min_total_verified=0)

        assert value == 0.0, f"GOAL-3 violated: empty targets should yield 0.0, got {value}"

    # -------------------------------------------------------------------------
    # GOAL-T4: Disjoint sets yield zero hits (GOAL-4)
    # -------------------------------------------------------------------------
    @l1
    def test_GOAL_T4_disjoint_sets(self):
        """GOAL-T4: verified ∩ targets = {} -> value=0.0 [GOAL-4]"""
        statements = StatementFactory.make_range("v", 20)  # v0..v19
        targets = StatementFactory.make_targets("t", 20)   # t0..t19 (disjoint)

        _, value = compute_goal_hit(statements, targets, min_total_verified=0)

        assert value == 0.0, f"GOAL-4 violated: disjoint sets should yield 0.0, got {value}"

    # -------------------------------------------------------------------------
    # GOAL-T5: Exact threshold yields success (GOAL-5)
    # -------------------------------------------------------------------------
    @l1
    def test_GOAL_T5_threshold_eq(self):
        """GOAL-T5: value == min_total_verified -> success=True [GOAL-5]"""
        statements = StatementFactory.make_statements(["t0", "t1", "t2"])
        targets = {"t0", "t1", "t2"}

        success, value = compute_goal_hit(statements, targets, min_total_verified=3)

        assert value == 3.0
        assert success is True, f"GOAL-5 violated: value=threshold should succeed"

    # -------------------------------------------------------------------------
    # GOAL-T6: Below threshold yields failure (GOAL-5)
    # -------------------------------------------------------------------------
    @l1
    def test_GOAL_T6_threshold_lt(self):
        """GOAL-T6: value < min_total_verified -> success=False [GOAL-5]"""
        statements = StatementFactory.make_statements(["t0", "t1"])
        targets = {"t0", "t1", "t2"}

        success, value = compute_goal_hit(statements, targets, min_total_verified=3)

        assert value == 2.0
        assert success is False, f"GOAL-5 violated: value<threshold should fail"

    # -------------------------------------------------------------------------
    # GOAL-T7: Above threshold yields success (GOAL-5)
    # -------------------------------------------------------------------------
    @l1
    def test_GOAL_T7_threshold_gt(self):
        """GOAL-T7: value > min_total_verified -> success=True [GOAL-5]"""
        statements = StatementFactory.make_statements(["t0", "t1", "t2", "t3"])
        targets = {"t0", "t1", "t2", "t3"}

        success, value = compute_goal_hit(statements, targets, min_total_verified=2)

        assert value == 4.0
        assert success is True, f"GOAL-5 violated: value>threshold should succeed"

    # -------------------------------------------------------------------------
    # GOAL-T8: Deterministic output (GOAL-6, Axiom D)
    # -------------------------------------------------------------------------
    @l0
    @determinism
    def test_GOAL_T8_determinism(self):
        """GOAL-T8: Same inputs yield identical outputs [GOAL-6, D]"""
        statements = StatementFactory.make_range("h", 50)
        targets = StatementFactory.make_targets("h", 20)

        results = [
            compute_goal_hit(statements, targets, min_total_verified=10)
            for _ in range(100)
        ]

        assert all(r == results[0] for r in results), "GOAL-6 violated: non-deterministic"

    # -------------------------------------------------------------------------
    # GOAL-T9: Order-independent (GOAL-7, Axiom O)
    # -------------------------------------------------------------------------
    @l1
    def test_GOAL_T9_order_independent(self):
        """GOAL-T9: Permuted verified list yields identical result [GOAL-7, O]"""
        targets = {"h1", "h3", "h5"}

        orderings = [
            StatementFactory.make_statements(["h0", "h1", "h2", "h3", "h4", "h5"]),
            StatementFactory.make_statements(["h5", "h4", "h3", "h2", "h1", "h0"]),
            StatementFactory.make_statements(["h3", "h1", "h5", "h0", "h4", "h2"]),
        ]

        results = [compute_goal_hit(o, targets, 2) for o in orderings]

        assert all(r == results[0] for r in results), "GOAL-7 violated: order-dependent"

    # -------------------------------------------------------------------------
    # GOAL-T10: Empty verified baseline (NC-1)
    # -------------------------------------------------------------------------
    @l1
    def test_GOAL_T10_empty_verified(self):
        """GOAL-T10: verified=[] -> value=0.0 [NC-1]"""
        targets = {"t0", "t1"}

        _, value = compute_goal_hit([], targets, min_total_verified=0)

        assert value == 0.0, f"NC-1 violated: empty verified should yield 0.0"

    # -------------------------------------------------------------------------
    # GOAL-T11: Impossibility detection (NC-2)
    # -------------------------------------------------------------------------
    @l1
    def test_GOAL_T11_impossibility(self):
        """GOAL-T11: Cannot exceed verified count [NC-2]"""
        statements = StatementFactory.make_statements(["t0", "t1", "t2"])
        targets = StatementFactory.make_targets("t", 100)  # 100 targets, only 3 verified

        _, value = compute_goal_hit(statements, targets, min_total_verified=0)

        assert value <= 3.0, f"NC-2 violated: value {value} exceeds verified count 3"

    # -------------------------------------------------------------------------
    # GOAL-T12: Threshold boundary behavior (NC-3)
    # -------------------------------------------------------------------------
    @l2
    @boundary
    def test_GOAL_T12_boundary(self):
        """GOAL-T12: T-1, T, T+1 boundary behavior [NC-3]"""
        statements = StatementFactory.make_statements(["t0", "t1", "t2"])
        targets = {"t0", "t1", "t2"}

        # T-1 (threshold=4, value=3)
        success_t_minus_1, _ = compute_goal_hit(statements, targets, min_total_verified=4)
        # T (threshold=3, value=3)
        success_t, _ = compute_goal_hit(statements, targets, min_total_verified=3)
        # T+1 (threshold=2, value=3)
        success_t_plus_1, _ = compute_goal_hit(statements, targets, min_total_verified=2)

        assert success_t_minus_1 is False, "NC-3 violated: T-1 should fail"
        assert success_t is True, "NC-3 violated: T should succeed"
        assert success_t_plus_1 is True, "NC-3 violated: T+1 should succeed"

    # -------------------------------------------------------------------------
    # GOAL-T13: Additive monotonicity (MON-1)
    # -------------------------------------------------------------------------
    @l2
    def test_GOAL_T13_additive_monotonicity(self):
        """GOAL-T13: V1 ⊆ V2 -> value(V1) <= value(V2) [MON-1]"""
        targets = {"t0", "t1", "t2", "t3", "t4"}

        v1 = StatementFactory.make_statements(["t0"])
        v2 = StatementFactory.make_statements(["t0", "t1"])
        v3 = StatementFactory.make_statements(["t0", "t1", "t2"])

        _, val1 = compute_goal_hit(v1, targets, 0)
        _, val2 = compute_goal_hit(v2, targets, 0)
        _, val3 = compute_goal_hit(v3, targets, 0)

        assert val1 <= val2 <= val3, f"MON-1 violated: {val1} <= {val2} <= {val3}"

    # -------------------------------------------------------------------------
    # GOAL-T14: Return type contract (Type)
    # -------------------------------------------------------------------------
    @l0
    @type_stability
    def test_GOAL_T14_return_type(self):
        """GOAL-T14: Returns Tuple[bool, float] [Type]"""
        result = compute_goal_hit(
            StatementFactory.make_statements(["h1"]),
            {"h1"},
            min_total_verified=1
        )

        assert isinstance(result, tuple), "Type violated: not a tuple"
        assert len(result) == 2, "Type violated: tuple length != 2"
        assert isinstance(result[0], bool), "Type violated: result[0] not bool"
        assert isinstance(result[1], float), "Type violated: result[1] not float"

    # -------------------------------------------------------------------------
    # GOAL-T15: Purity (Axiom P)
    # -------------------------------------------------------------------------
    @l1
    def test_GOAL_T15_purity(self):
        """GOAL-T15: No side effects [P]"""
        statements = StatementFactory.make_range("h", 10)
        targets = {"h1", "h2", "h3"}
        statements_copy = [dict(s) for s in statements]
        targets_copy = set(targets)

        compute_goal_hit(statements, targets, min_total_verified=1)

        # Verify inputs unchanged
        assert statements == statements_copy, "P violated: statements mutated"
        assert targets == targets_copy, "P violated: targets mutated"


# =============================================================================
# SPARSE_SUCCESS CONFORMANCE TESTS (SPARSE-T1 through SPARSE-T14)
# =============================================================================

class TestSparseSuccessConformance:
    """
    Conformance tests for compute_sparse_success.
    Reference: METRIC_CONFORMANCE_SUITE_SPEC.md Section 4.
    """

    # -------------------------------------------------------------------------
    # SPARSE-T1: Value equals input (SPARSE-1)
    # -------------------------------------------------------------------------
    @l1
    def test_SPARSE_T1_value_equals_input(self):
        """SPARSE-T1: value == float(verified_count) [SPARSE-1]"""
        for v in [0, 1, 5, 10, 50, 100, 1000]:
            _, value = compute_sparse_success(v, 999, min_verified=0)
            assert value == float(v), f"SPARSE-1 violated: expected {float(v)}, got {value}"

    # -------------------------------------------------------------------------
    # SPARSE-T2: Exact threshold succeeds (SPARSE-2)
    # -------------------------------------------------------------------------
    @l1
    def test_SPARSE_T2_threshold_eq(self):
        """SPARSE-T2: verified == min_verified -> success=True [SPARSE-2]"""
        success, _ = compute_sparse_success(5, 10, min_verified=5)
        assert success is True, "SPARSE-2 violated: exact threshold should succeed"

    # -------------------------------------------------------------------------
    # SPARSE-T3: Below threshold fails (SPARSE-2)
    # -------------------------------------------------------------------------
    @l1
    def test_SPARSE_T3_threshold_lt(self):
        """SPARSE-T3: verified < min_verified -> success=False [SPARSE-2]"""
        success, _ = compute_sparse_success(4, 10, min_verified=5)
        assert success is False, "SPARSE-2 violated: below threshold should fail"

    # -------------------------------------------------------------------------
    # SPARSE-T4: Above threshold succeeds (SPARSE-2)
    # -------------------------------------------------------------------------
    @l1
    def test_SPARSE_T4_threshold_gt(self):
        """SPARSE-T4: verified > min_verified -> success=True [SPARSE-2]"""
        success, _ = compute_sparse_success(6, 10, min_verified=5)
        assert success is True, "SPARSE-2 violated: above threshold should succeed"

    # -------------------------------------------------------------------------
    # SPARSE-T5: Attempted count ignored (SPARSE-3)
    # -------------------------------------------------------------------------
    @l1
    def test_SPARSE_T5_attempted_ignored(self):
        """SPARSE-T5: compute(v, a1, m) == compute(v, a2, m) [SPARSE-3]"""
        verified = 7
        min_v = 5

        results = [
            compute_sparse_success(verified, attempted, min_v)
            for attempted in [0, 1, 10, 100, 1000, 999999]
        ]

        assert all(r == results[0] for r in results), "SPARSE-3 violated: attempted affects result"

    # -------------------------------------------------------------------------
    # SPARSE-T6: Zero verified yields zero (SPARSE-4)
    # -------------------------------------------------------------------------
    @l1
    def test_SPARSE_T6_zero_verified(self):
        """SPARSE-T6: verified_count=0 -> value=0.0 [SPARSE-4]"""
        _, value = compute_sparse_success(0, 100, min_verified=0)
        assert value == 0.0, f"SPARSE-4 violated: zero verified should yield 0.0"

    # -------------------------------------------------------------------------
    # SPARSE-T7: Value non-negative (SPARSE-5)
    # -------------------------------------------------------------------------
    @l1
    def test_SPARSE_T7_non_negative(self):
        """SPARSE-T7: value >= 0.0 [SPARSE-5]"""
        for v in range(100):
            _, value = compute_sparse_success(v, 100, min_verified=0)
            assert value >= 0.0, f"SPARSE-5 violated: value {value} < 0"

    # -------------------------------------------------------------------------
    # SPARSE-T8: Deterministic output (SPARSE-6, Axiom D)
    # -------------------------------------------------------------------------
    @l0
    @determinism
    def test_SPARSE_T8_determinism(self):
        """SPARSE-T8: Same inputs yield identical outputs [SPARSE-6, D]"""
        results = [
            compute_sparse_success(42, 100, min_verified=30)
            for _ in range(100)
        ]

        assert all(r == results[0] for r in results), "SPARSE-6 violated: non-deterministic"

    # -------------------------------------------------------------------------
    # SPARSE-T9: Zero input baseline (NC-1)
    # -------------------------------------------------------------------------
    @l1
    def test_SPARSE_T9_zero_input_baseline(self):
        """SPARSE-T9: verified=0, min=1 -> success=False [NC-1]"""
        success, value = compute_sparse_success(0, 100, min_verified=1)
        assert success is False, "NC-1 violated: zero input with threshold should fail"
        assert value == 0.0

    # -------------------------------------------------------------------------
    # SPARSE-T10: Threshold boundary behavior (NC-3)
    # -------------------------------------------------------------------------
    @l2
    @boundary
    def test_SPARSE_T10_boundary(self):
        """SPARSE-T10: T-1, T, T+1 boundary behavior [NC-3]"""
        # T-1
        success_t_minus_1, _ = compute_sparse_success(4, 10, min_verified=5)
        # T
        success_t, _ = compute_sparse_success(5, 10, min_verified=5)
        # T+1
        success_t_plus_1, _ = compute_sparse_success(6, 10, min_verified=5)

        assert success_t_minus_1 is False, "NC-3 violated: T-1 should fail"
        assert success_t is True, "NC-3 violated: T should succeed"
        assert success_t_plus_1 is True, "NC-3 violated: T+1 should succeed"

    # -------------------------------------------------------------------------
    # SPARSE-T11: Additive monotonicity (MON-1)
    # -------------------------------------------------------------------------
    @l2
    def test_SPARSE_T11_additive_monotonicity(self):
        """SPARSE-T11: V1 < V2 -> value(V1) < value(V2) [MON-1]"""
        _, val1 = compute_sparse_success(5, 100, 0)
        _, val2 = compute_sparse_success(10, 100, 0)
        _, val3 = compute_sparse_success(15, 100, 0)

        assert val1 < val2 < val3, f"MON-1 violated: {val1} < {val2} < {val3}"

    # -------------------------------------------------------------------------
    # SPARSE-T12: Threshold monotonicity (MON-2)
    # -------------------------------------------------------------------------
    @l2
    def test_SPARSE_T12_threshold_monotonicity(self):
        """SPARSE-T12: Lower threshold easier [MON-2]"""
        # With verified=5, test different thresholds
        s_t3, _ = compute_sparse_success(5, 10, min_verified=3)
        s_t5, _ = compute_sparse_success(5, 10, min_verified=5)
        s_t7, _ = compute_sparse_success(5, 10, min_verified=7)

        # Lower threshold should have same or higher success rate
        assert s_t3 is True
        assert s_t5 is True
        assert s_t7 is False

    # -------------------------------------------------------------------------
    # SPARSE-T13: Return type contract (Type)
    # -------------------------------------------------------------------------
    @l0
    @type_stability
    def test_SPARSE_T13_return_type(self):
        """SPARSE-T13: Returns Tuple[bool, float] [Type]"""
        result = compute_sparse_success(5, 10, min_verified=3)

        assert isinstance(result, tuple), "Type violated: not a tuple"
        assert len(result) == 2, "Type violated: tuple length != 2"
        assert isinstance(result[0], bool), "Type violated: result[0] not bool"
        assert isinstance(result[1], float), "Type violated: result[1] not float"

    # -------------------------------------------------------------------------
    # SPARSE-T14: Purity (Axiom P)
    # -------------------------------------------------------------------------
    @l1
    def test_SPARSE_T14_purity(self):
        """SPARSE-T14: No side effects [P]"""
        # For scalar inputs, purity is trivially satisfied
        # This test verifies the function doesn't modify any global state
        import sys
        frame_count_before = len(sys._current_frames())

        compute_sparse_success(10, 20, 5)

        frame_count_after = len(sys._current_frames())
        # Frame count should be approximately the same (may vary slightly due to GC)
        assert abs(frame_count_after - frame_count_before) <= 1, "P potentially violated"


# =============================================================================
# CHAIN_SUCCESS CONFORMANCE TESTS (CHAIN-T1 through CHAIN-T17)
# =============================================================================

class TestChainSuccessConformance:
    """
    Conformance tests for compute_chain_success.
    Reference: METRIC_CONFORMANCE_SUITE_SPEC.md Section 5.
    """

    # -------------------------------------------------------------------------
    # CHAIN-T1: Value bounded by verified count (CHAIN-1)
    # -------------------------------------------------------------------------
    @l1
    def test_CHAIN_T1_verified_bound(self):
        """CHAIN-T1: value <= len(verified_statements) [CHAIN-1]"""
        graph = GraphFactory.make_linear_chain(10)
        statements = StatementFactory.make_range("h", 5)  # Only 5 verified

        _, value = compute_chain_success(statements, graph, "h4", min_chain_length=1)

        assert value <= len(statements), f"CHAIN-1 violated: value {value} > verified {len(statements)}"

    # -------------------------------------------------------------------------
    # CHAIN-T2: Unverified target yields zero (CHAIN-2)
    # -------------------------------------------------------------------------
    @l1
    def test_CHAIN_T2_unverified_target(self):
        """CHAIN-T2: target not in verified -> value=0.0 [CHAIN-2]"""
        graph = {"h2": ["h1"], "h1": ["h0"]}
        statements = StatementFactory.make_statements(["h0", "h1"])  # h2 not verified

        _, value = compute_chain_success(statements, graph, "h2", min_chain_length=1)

        assert value == 0.0, f"CHAIN-2 violated: unverified target should yield 0.0"

    # -------------------------------------------------------------------------
    # CHAIN-T3: Isolated target yields one (CHAIN-3)
    # -------------------------------------------------------------------------
    @l1
    def test_CHAIN_T3_isolated_target(self):
        """CHAIN-T3: target in verified, deps={} -> value=1.0 [CHAIN-3]"""
        graph = {}  # No dependencies
        statements = StatementFactory.make_statements(["isolated"])

        _, value = compute_chain_success(statements, graph, "isolated", min_chain_length=1)

        assert value == 1.0, f"CHAIN-3 violated: isolated target should yield 1.0"

    # -------------------------------------------------------------------------
    # CHAIN-T4: Cycle safety (CHAIN-4)
    # -------------------------------------------------------------------------
    @l1
    def test_CHAIN_T4_cycle_safe(self):
        """CHAIN-T4: Cyclic graph does not cause RecursionError [CHAIN-4]"""
        graph = GraphFactory.make_cycle(["h0", "h1", "h2"])
        statements = StatementFactory.make_statements(["h0", "h1", "h2"])

        try:
            compute_chain_success(statements, graph, "h0", min_chain_length=1)
        except RecursionError:
            pytest.fail("CHAIN-4 violated: cycle caused infinite recursion")

    # -------------------------------------------------------------------------
    # CHAIN-T5: Exact threshold succeeds (CHAIN-5)
    # -------------------------------------------------------------------------
    @l1
    def test_CHAIN_T5_threshold_eq(self):
        """CHAIN-T5: value == min_chain_length -> success=True [CHAIN-5]"""
        graph = {"h2": ["h1"], "h1": ["h0"]}
        statements = StatementFactory.make_statements(["h0", "h1", "h2"])

        success, value = compute_chain_success(statements, graph, "h2", min_chain_length=3)

        assert value == 3.0
        assert success is True, "CHAIN-5 violated: exact threshold should succeed"

    # -------------------------------------------------------------------------
    # CHAIN-T6: Below threshold fails (CHAIN-5)
    # -------------------------------------------------------------------------
    @l1
    def test_CHAIN_T6_threshold_lt(self):
        """CHAIN-T6: value < min_chain_length -> success=False [CHAIN-5]"""
        graph = {"h2": ["h1"], "h1": ["h0"]}
        statements = StatementFactory.make_statements(["h0", "h1", "h2"])

        success, value = compute_chain_success(statements, graph, "h2", min_chain_length=4)

        assert value == 3.0
        assert success is False, "CHAIN-5 violated: below threshold should fail"

    # -------------------------------------------------------------------------
    # CHAIN-T7: Above threshold succeeds (CHAIN-5)
    # -------------------------------------------------------------------------
    @l1
    def test_CHAIN_T7_threshold_gt(self):
        """CHAIN-T7: value > min_chain_length -> success=True [CHAIN-5]"""
        graph = {"h2": ["h1"], "h1": ["h0"]}
        statements = StatementFactory.make_statements(["h0", "h1", "h2"])

        success, value = compute_chain_success(statements, graph, "h2", min_chain_length=2)

        assert value == 3.0
        assert success is True, "CHAIN-5 violated: above threshold should succeed"

    # -------------------------------------------------------------------------
    # CHAIN-T8: Longest path selected (CHAIN-6)
    # -------------------------------------------------------------------------
    @l1
    def test_CHAIN_T8_longest_path(self):
        """CHAIN-T8: Diamond graph selects longest path [CHAIN-6]"""
        graph = GraphFactory.make_diamond()
        statements = StatementFactory.make_statements(["h0", "h1", "h2", "h3"])

        _, value = compute_chain_success(statements, graph, "h3", min_chain_length=1)

        # Longest path is h3 -> h1 -> h0 or h3 -> h2 -> h0 = 3
        assert value == 3.0, f"CHAIN-6 violated: longest path should be 3, got {value}"

    # -------------------------------------------------------------------------
    # CHAIN-T9: Deterministic output (CHAIN-7, Axiom D)
    # -------------------------------------------------------------------------
    @l0
    @determinism
    def test_CHAIN_T9_determinism(self):
        """CHAIN-T9: Same inputs yield identical outputs [CHAIN-7, D]"""
        graph = GraphFactory.make_linear_chain(10)
        statements = StatementFactory.make_range("h", 10)

        results = [
            compute_chain_success(statements, graph, "h9", min_chain_length=5)
            for _ in range(100)
        ]

        assert all(r == results[0] for r in results), "CHAIN-7 violated: non-deterministic"

    # -------------------------------------------------------------------------
    # CHAIN-T10: Order-independent (CHAIN-8, Axiom O)
    # -------------------------------------------------------------------------
    @l1
    def test_CHAIN_T10_order_independent(self):
        """CHAIN-T10: Permuted verified list yields identical result [CHAIN-8, O]"""
        graph = {"h2": ["h1"], "h1": ["h0"]}

        orderings = [
            StatementFactory.make_statements(["h0", "h1", "h2"]),
            StatementFactory.make_statements(["h2", "h1", "h0"]),
            StatementFactory.make_statements(["h1", "h0", "h2"]),
        ]

        results = [compute_chain_success(o, graph, "h2", 3) for o in orderings]

        assert all(r == results[0] for r in results), "CHAIN-8 violated: order-dependent"

    # -------------------------------------------------------------------------
    # CHAIN-T11: Empty verified baseline (NC-1)
    # -------------------------------------------------------------------------
    @l1
    def test_CHAIN_T11_empty_verified(self):
        """CHAIN-T11: verified=[] -> value=0.0 [NC-1]"""
        graph = {"h2": ["h1"], "h1": ["h0"]}

        _, value = compute_chain_success([], graph, "h2", min_chain_length=1)

        assert value == 0.0, f"NC-1 violated: empty verified should yield 0.0"

    # -------------------------------------------------------------------------
    # CHAIN-T12: Impossibility detection (NC-2)
    # -------------------------------------------------------------------------
    @l1
    def test_CHAIN_T12_impossibility(self):
        """CHAIN-T12: Cannot exceed node count [NC-2]"""
        graph = GraphFactory.make_linear_chain(10)
        statements = StatementFactory.make_range("h", 3)  # Only 3 verified

        success, value = compute_chain_success(statements, graph, "h2", min_chain_length=5)

        assert success is False, "NC-2 violated: should fail when chain impossible"
        assert value <= 3.0, f"NC-2 violated: value {value} > verified count 3"

    # -------------------------------------------------------------------------
    # CHAIN-T13: Threshold boundary behavior (NC-3)
    # -------------------------------------------------------------------------
    @l2
    @boundary
    def test_CHAIN_T13_boundary(self):
        """CHAIN-T13: T-1, T, T+1 boundary behavior [NC-3]"""
        graph = {"h2": ["h1"], "h1": ["h0"]}
        statements = StatementFactory.make_statements(["h0", "h1", "h2"])
        # Chain length = 3

        # T-1 (threshold=4)
        success_t_minus_1, _ = compute_chain_success(statements, graph, "h2", min_chain_length=4)
        # T (threshold=3)
        success_t, _ = compute_chain_success(statements, graph, "h2", min_chain_length=3)
        # T+1 (threshold=2)
        success_t_plus_1, _ = compute_chain_success(statements, graph, "h2", min_chain_length=2)

        assert success_t_minus_1 is False, "NC-3 violated: T-1 should fail"
        assert success_t is True, "NC-3 violated: T should succeed"
        assert success_t_plus_1 is True, "NC-3 violated: T+1 should succeed"

    # -------------------------------------------------------------------------
    # CHAIN-T14: Return type contract (Type)
    # -------------------------------------------------------------------------
    @l0
    @type_stability
    def test_CHAIN_T14_return_type(self):
        """CHAIN-T14: Returns Tuple[bool, float] [Type]"""
        result = compute_chain_success(
            StatementFactory.make_statements(["h1"]),
            {},
            "h1",
            min_chain_length=1
        )

        assert isinstance(result, tuple), "Type violated: not a tuple"
        assert len(result) == 2, "Type violated: tuple length != 2"
        assert isinstance(result[0], bool), "Type violated: result[0] not bool"
        assert isinstance(result[1], float), "Type violated: result[1] not float"

    # -------------------------------------------------------------------------
    # CHAIN-T15: Purity (Axiom P)
    # -------------------------------------------------------------------------
    @l1
    def test_CHAIN_T15_purity(self):
        """CHAIN-T15: No side effects [P]"""
        statements = StatementFactory.make_range("h", 5)
        graph = GraphFactory.make_linear_chain(5)
        statements_copy = [dict(s) for s in statements]
        graph_copy = {k: list(v) for k, v in graph.items()}

        compute_chain_success(statements, graph, "h4", min_chain_length=3)

        assert statements == statements_copy, "P violated: statements mutated"
        assert graph == graph_copy, "P violated: graph mutated"

    # -------------------------------------------------------------------------
    # CHAIN-T16: Deep cycle safety (CHAIN-4)
    # -------------------------------------------------------------------------
    @l2
    def test_CHAIN_T16_deep_cycle(self):
        """CHAIN-T16: Self-loop graph completes [CHAIN-4]"""
        # Self-loop: node depends on itself
        graph = {"h0": ["h0"]}
        statements = StatementFactory.make_statements(["h0"])

        try:
            result = compute_chain_success(statements, graph, "h0", min_chain_length=1)
            assert result[0] is True  # Should succeed (node is verified)
        except RecursionError:
            pytest.fail("CHAIN-4 violated: self-loop caused infinite recursion")

    # -------------------------------------------------------------------------
    # CHAIN-T17: Broken chain measurement (CHAIN-6)
    # -------------------------------------------------------------------------
    @l2
    def test_CHAIN_T17_broken_chain(self):
        """CHAIN-T17: Gap in chain terminates correctly [CHAIN-6]"""
        graph = {"h3": ["h2"], "h2": ["h1"], "h1": ["h0"]}
        # Gap: h1 not verified
        statements = StatementFactory.make_statements(["h0", "h2", "h3"])

        _, value = compute_chain_success(statements, graph, "h3", min_chain_length=1)

        # Chain should be: h3 -> h2 (h1 missing) = 2
        assert value == 2.0, f"CHAIN-6 violated: broken chain should yield 2, got {value}"


# =============================================================================
# MULTI_GOAL_SUCCESS CONFORMANCE TESTS (MULTI-T1 through MULTI-T15)
# =============================================================================

class TestMultiGoalSuccessConformance:
    """
    Conformance tests for compute_multi_goal_success.
    Reference: METRIC_CONFORMANCE_SUITE_SPEC.md Section 6.
    """

    # -------------------------------------------------------------------------
    # MULTI-T1: Value bounded by required count (MULTI-1)
    # -------------------------------------------------------------------------
    @l1
    def test_MULTI_T1_required_bound(self):
        """MULTI-T1: value <= len(required_goal_hashes) [MULTI-1]"""
        verified = StatementFactory.make_targets("h", 100)
        required = {"r0", "r1", "r2"}  # 3 required

        _, value = compute_multi_goal_success(verified, required)

        assert value <= len(required), f"MULTI-1 violated: value {value} > required {len(required)}"

    # -------------------------------------------------------------------------
    # MULTI-T2: All goals met yields success (MULTI-2)
    # -------------------------------------------------------------------------
    @l1
    def test_MULTI_T2_all_met(self):
        """MULTI-T2: verified ⊇ required -> success=True [MULTI-2]"""
        verified = {"g0", "g1", "g2", "g3", "extra"}
        required = {"g0", "g1", "g2"}

        success, value = compute_multi_goal_success(verified, required)

        assert value == 3.0
        assert success is True, "MULTI-2 violated: all goals met should succeed"

    # -------------------------------------------------------------------------
    # MULTI-T3: Partial goals yield failure (MULTI-2)
    # -------------------------------------------------------------------------
    @l1
    def test_MULTI_T3_partial(self):
        """MULTI-T3: verified ∩ required ≠ required -> success=False [MULTI-2]"""
        verified = {"g0", "g1"}  # Missing g2
        required = {"g0", "g1", "g2"}

        success, value = compute_multi_goal_success(verified, required)

        assert value == 2.0
        assert success is False, "MULTI-2 violated: partial goals should fail"

    # -------------------------------------------------------------------------
    # MULTI-T4: No goals met (MULTI-2)
    # -------------------------------------------------------------------------
    @l1
    def test_MULTI_T4_none_met(self):
        """MULTI-T4: verified ∩ required = {} -> success=False, value=0.0 [MULTI-2]"""
        verified = {"other1", "other2"}
        required = {"g0", "g1", "g2"}

        success, value = compute_multi_goal_success(verified, required)

        assert value == 0.0
        assert success is False, "MULTI-2 violated: no goals met should fail"

    # -------------------------------------------------------------------------
    # MULTI-T5: Empty required is vacuous truth (MULTI-3)
    # -------------------------------------------------------------------------
    @l1
    def test_MULTI_T5_empty_required(self):
        """MULTI-T5: required={} -> success=True, value=0.0 [MULTI-3]

        Per METRIC_CORRECTNESS_CONTRACT.md, empty required should return
        success=True (vacuous truth - all 0 required goals are satisfied).
        """
        verified = {"h1", "h2", "h3"}

        success, value = compute_multi_goal_success(verified, set())

        # Empty required means vacuous truth -> success
        assert success is True, "Empty required = vacuous truth = success"
        assert value == 0.0, "Empty required should yield 0.0"

    # -------------------------------------------------------------------------
    # MULTI-T6: Subset monotonicity (MULTI-4)
    # -------------------------------------------------------------------------
    @l2
    def test_MULTI_T6_subset_monotonicity(self):
        """MULTI-T6: V1 ⊆ V2 -> value(V1) <= value(V2) [MULTI-4]"""
        required = {"g0", "g1", "g2", "g3", "g4"}

        v1 = {"g0", "g1"}
        v2 = {"g0", "g1", "g2"}
        v3 = {"g0", "g1", "g2", "g3", "g4", "extra"}

        _, val1 = compute_multi_goal_success(v1, required)
        _, val2 = compute_multi_goal_success(v2, required)
        _, val3 = compute_multi_goal_success(v3, required)

        assert val1 <= val2 <= val3, f"MULTI-4 violated: {val1} <= {val2} <= {val3}"

    # -------------------------------------------------------------------------
    # MULTI-T7: Correct intersection counting (MULTI-5)
    # -------------------------------------------------------------------------
    @l1
    def test_MULTI_T7_counting(self):
        """MULTI-T7: value == len(verified ∩ required) [MULTI-5]"""
        verified = {"a", "b", "c", "d", "e"}
        required = {"b", "d", "f", "h"}

        _, value = compute_multi_goal_success(verified, required)

        expected = len(verified & required)  # {"b", "d"} = 2
        assert value == float(expected), f"MULTI-5 violated: expected {expected}, got {value}"

    # -------------------------------------------------------------------------
    # MULTI-T8: Deterministic output (MULTI-6, Axiom D)
    # -------------------------------------------------------------------------
    @l0
    @determinism
    def test_MULTI_T8_determinism(self):
        """MULTI-T8: Same inputs yield identical outputs [MULTI-6, D]"""
        verified = StatementFactory.make_targets("v", 50)
        required = StatementFactory.make_targets("v", 20)

        results = [
            compute_multi_goal_success(verified, required)
            for _ in range(100)
        ]

        assert all(r == results[0] for r in results), "MULTI-6 violated: non-deterministic"

    # -------------------------------------------------------------------------
    # MULTI-T9: Empty verified baseline (NC-1)
    # -------------------------------------------------------------------------
    @l1
    def test_MULTI_T9_empty_verified(self):
        """MULTI-T9: verified={}, required={r1} -> success=False [NC-1]"""
        success, value = compute_multi_goal_success(set(), {"r1"})

        assert success is False, "NC-1 violated: empty verified with required should fail"
        assert value == 0.0

    # -------------------------------------------------------------------------
    # MULTI-T10: Impossibility detection (NC-2)
    # -------------------------------------------------------------------------
    @l1
    def test_MULTI_T10_impossibility(self):
        """MULTI-T10: len(verified) < len(required) -> value <= len(verified) [NC-2]"""
        verified = {"g0"}  # 1 verified
        required = {"g0", "g1", "g2", "g3"}  # 4 required

        _, value = compute_multi_goal_success(verified, required)

        assert value <= len(verified), f"NC-2 violated: value {value} > verified {len(verified)}"

    # -------------------------------------------------------------------------
    # MULTI-T11: Additive monotonicity (MON-1)
    # -------------------------------------------------------------------------
    @l2
    def test_MULTI_T11_additive_monotonicity(self):
        """MULTI-T11: V1 ⊆ V2 -> value(V1) <= value(V2) [MON-1]"""
        required = {"g0", "g1", "g2", "g3", "g4"}

        v1 = {"g0"}
        v2 = {"g0", "g1"}
        v3 = {"g0", "g1", "g2", "g3", "g4"}

        _, val1 = compute_multi_goal_success(v1, required)
        _, val2 = compute_multi_goal_success(v2, required)
        _, val3 = compute_multi_goal_success(v3, required)

        assert val1 <= val2 <= val3, f"MON-1 violated: {val1} <= {val2} <= {val3}"

    # -------------------------------------------------------------------------
    # MULTI-T12: Subset required monotonicity (MON-3)
    # -------------------------------------------------------------------------
    @l2
    def test_MULTI_T12_subset_required(self):
        """MULTI-T12: R1 ⊆ R2, success(V, R2) -> success(V, R1) [MON-3]"""
        verified = {"g0", "g1", "g2", "g3", "g4"}
        r1 = {"g0", "g1"}  # Subset
        r2 = {"g0", "g1", "g2", "g3", "g4"}  # Superset

        s1, _ = compute_multi_goal_success(verified, r1)
        s2, _ = compute_multi_goal_success(verified, r2)

        # If success with larger required set, must succeed with smaller
        if s2:
            assert s1 is True, "MON-3 violated: success(V, R2) should imply success(V, R1)"

    # -------------------------------------------------------------------------
    # MULTI-T13: Return type contract (Type)
    # -------------------------------------------------------------------------
    @l0
    @type_stability
    def test_MULTI_T13_return_type(self):
        """MULTI-T13: Returns Tuple[bool, float] [Type]"""
        result = compute_multi_goal_success({"h1"}, {"h1"})

        assert isinstance(result, tuple), "Type violated: not a tuple"
        assert len(result) == 2, "Type violated: tuple length != 2"
        assert isinstance(result[0], bool), "Type violated: result[0] not bool"
        assert isinstance(result[1], float), "Type violated: result[1] not float"

    # -------------------------------------------------------------------------
    # MULTI-T14: Purity (Axiom P)
    # -------------------------------------------------------------------------
    @l1
    def test_MULTI_T14_purity(self):
        """MULTI-T14: No side effects [P]"""
        verified = {"a", "b", "c"}
        required = {"b", "c", "d"}
        verified_copy = set(verified)
        required_copy = set(required)

        compute_multi_goal_success(verified, required)

        assert verified == verified_copy, "P violated: verified mutated"
        assert required == required_copy, "P violated: required mutated"

    # -------------------------------------------------------------------------
    # MULTI-T15: Order independence (Axiom O)
    # -------------------------------------------------------------------------
    @l1
    def test_MULTI_T15_order_independent(self):
        """MULTI-T15: Set operations are inherently order-independent [O]"""
        # Sets are unordered, but we verify consistent results
        verified = {"z", "a", "m", "b", "y"}
        required = {"a", "b", "c"}

        results = [
            compute_multi_goal_success(verified, required)
            for _ in range(10)
        ]

        assert all(r == results[0] for r in results), "O violated: inconsistent results"


# =============================================================================
# STRESS TESTS (L3 EXHAUSTIVE)
# =============================================================================

class TestConformanceStress:
    """
    Stress tests for L3 exhaustive conformance level.
    These run 1000+ iterations to verify determinism under load.
    """

    @l3
    @determinism
    def test_STRESS_goal_hit_1000_iterations(self):
        """Stress test: goal_hit determinism over 1000 iterations"""
        statements = StatementFactory.make_range("h", 100)
        targets = StatementFactory.make_targets("h", 50)

        results = [
            compute_goal_hit(statements, targets, min_total_verified=25)
            for _ in range(1000)
        ]

        assert all(r == results[0] for r in results)

    @l3
    @determinism
    def test_STRESS_sparse_success_1000_iterations(self):
        """Stress test: sparse_success determinism over 1000 iterations"""
        results = [
            compute_sparse_success(75, 100, min_verified=50)
            for _ in range(1000)
        ]

        assert all(r == results[0] for r in results)

    @l3
    @determinism
    def test_STRESS_chain_success_1000_iterations(self):
        """Stress test: chain_success determinism over 1000 iterations"""
        graph = GraphFactory.make_linear_chain(50)
        statements = StatementFactory.make_range("h", 50)

        results = [
            compute_chain_success(statements, graph, "h49", min_chain_length=25)
            for _ in range(1000)
        ]

        assert all(r == results[0] for r in results)

    @l3
    @determinism
    def test_STRESS_multi_goal_1000_iterations(self):
        """Stress test: multi_goal_success determinism over 1000 iterations"""
        verified = StatementFactory.make_targets("v", 200)
        required = StatementFactory.make_targets("v", 100)

        results = [
            compute_multi_goal_success(verified, required)
            for _ in range(1000)
        ]

        assert all(r == results[0] for r in results)

    @l3
    def test_STRESS_large_graph(self):
        """Stress test: chain_success with large graph (within recursion limit)"""
        # Use depth that stays within Python's default recursion limit (~1000)
        # A chain of 200 is reasonable and tests scale
        graph = GraphFactory.make_linear_chain(200)
        statements = StatementFactory.make_range("h", 200)

        success, value = compute_chain_success(statements, graph, "h199", min_chain_length=100)

        assert success is True
        assert value == 200.0


# =============================================================================
# PYTEST CONFIGURATION FOR CONFORMANCE LEVELS
# =============================================================================

def pytest_configure(config):
    """Register custom markers for conformance levels."""
    config.addinivalue_line(
        "markers", "conformance_l0: L0 minimal conformance (smoke tests)"
    )
    config.addinivalue_line(
        "markers", "conformance_l1: L1 standard conformance (all invariants)"
    )
    config.addinivalue_line(
        "markers", "conformance_l2: L2 full conformance (invariants + boundary + monotonicity)"
    )
    config.addinivalue_line(
        "markers", "conformance_l3: L3 exhaustive conformance (full + stress)"
    )
