# tests/phase2/metrics/conftest.py
"""
Phase II Metrics Test Battery - Shared Fixtures and Configuration

All fixtures are deterministic and use seeded random generators.
"""

import random
from typing import Any, Dict, List, Set, Tuple
import pytest


# ===========================================================================
# PRNG SEEDS - DO NOT CHANGE THESE VALUES
# ===========================================================================
# These seeds ensure deterministic behavior across all test runs.

SEED_BASE = 42
SEED_GOAL_HIT = 1001
SEED_SPARSE_DENSITY = 2002
SEED_CHAIN_LENGTH = 3003
SEED_MULTI_GOAL = 4004
SEED_ADAPTER = 5005
SEED_REPLAY = 6006


# ===========================================================================
# TYPE STABILITY ASSERTION HELPERS
# ===========================================================================

def assert_bool_type(value: Any, context: str = "") -> None:
    """Assert that value is strictly bool, not truthy/falsy."""
    assert isinstance(value, bool), f"Expected bool, got {type(value).__name__}. {context}"


def assert_float_type(value: Any, context: str = "") -> None:
    """Assert that value is strictly float."""
    assert isinstance(value, float), f"Expected float, got {type(value).__name__}. {context}"


def assert_int_type(value: Any, context: str = "") -> None:
    """Assert that value is strictly int."""
    assert isinstance(value, int), f"Expected int, got {type(value).__name__}. {context}"


def assert_tuple_bool_float(result: Any, context: str = "") -> None:
    """Assert result is tuple of (bool, float)."""
    assert isinstance(result, tuple), f"Expected tuple, got {type(result).__name__}. {context}"
    assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple. {context}"
    assert_bool_type(result[0], f"First element. {context}")
    assert_float_type(result[1], f"Second element. {context}")


def assert_no_uplift_interpretation() -> str:
    """
    Returns a docstring snippet asserting no uplift interpretation.
    This is a mechanical test - it verifies logic, not meaning.
    """
    return "NO UPLIFT INTERPRETATION: This test verifies mechanical correctness only."


# ===========================================================================
# DETERMINISTIC DATA GENERATORS
# ===========================================================================

class DeterministicGenerator:
    """
    Seeded generator for test data.
    
    All methods produce identical output for the same seed.
    """
    
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._seed = seed
    
    @property
    def seed(self) -> int:
        return self._seed
    
    def reset(self) -> None:
        """Reset to initial seed state."""
        self._rng = random.Random(self._seed)
    
    def hash(self, prefix: str = "h") -> str:
        """Generate a deterministic hash string."""
        return f"{prefix}{self._rng.randint(0, 99999):05d}"
    
    def hash_list(self, count: int, prefix: str = "h") -> List[str]:
        """Generate a list of deterministic hashes."""
        return [self.hash(prefix) for _ in range(count)]
    
    def hash_set(self, count: int, prefix: str = "h") -> Set[str]:
        """Generate a set of deterministic hashes."""
        result: Set[str] = set()
        while len(result) < count:
            result.add(self.hash(prefix))
        return result
    
    def statements(self, count: int, prefix: str = "h") -> List[Dict[str, Any]]:
        """Generate a list of statement dicts with hashes."""
        return [{"hash": h} for h in self.hash_list(count, prefix)]
    
    def statements_from_hashes(self, hashes: List[str]) -> List[Dict[str, Any]]:
        """Convert hash list to statement dicts."""
        return [{"hash": h} for h in hashes]
    
    def dependency_graph_linear(self, depth: int, prefix: str = "h") -> Tuple[Dict[str, List[str]], List[str]]:
        """
        Generate a linear dependency chain: h0 <- h1 <- h2 <- ... <- h(depth-1)
        Returns (graph, hashes_in_order)
        """
        hashes = [f"{prefix}{i}" for i in range(depth)]
        graph = {}
        for i in range(1, depth):
            graph[hashes[i]] = [hashes[i - 1]]
        return graph, hashes
    
    def dependency_graph_diamond(self, prefix: str = "h") -> Tuple[Dict[str, List[str]], List[str]]:
        """
        Generate a diamond dependency graph:
            h0
           /  \
          h1  h2
           \  /
            h3
        Returns (graph, hashes)
        """
        hashes = [f"{prefix}{i}" for i in range(4)]
        graph = {
            hashes[1]: [hashes[0]],
            hashes[2]: [hashes[0]],
            hashes[3]: [hashes[1], hashes[2]],
        }
        return graph, hashes
    
    def dependency_graph_random(
        self,
        num_nodes: int,
        edge_probability: float = 0.3,
        prefix: str = "h"
    ) -> Tuple[Dict[str, List[str]], List[str]]:
        """
        Generate a random DAG with given number of nodes and edge probability.
        Edges only go from higher to lower numbered nodes (ensures DAG property).
        """
        hashes = [f"{prefix}{i}" for i in range(num_nodes)]
        graph: Dict[str, List[str]] = {}
        
        for i in range(1, num_nodes):
            deps = []
            for j in range(i):
                if self._rng.random() < edge_probability:
                    deps.append(hashes[j])
            if deps:
                graph[hashes[i]] = deps
        
        return graph, hashes
    
    def float_value(self, min_val: float = 0.0, max_val: float = 10.0) -> float:
        """Generate a random float in range."""
        return self._rng.uniform(min_val, max_val)
    
    def float_list(self, count: int, min_val: float = 0.0, max_val: float = 10.0) -> List[float]:
        """Generate a list of random floats."""
        return [self.float_value(min_val, max_val) for _ in range(count)]
    
    def int_value(self, min_val: int = 0, max_val: int = 100) -> int:
        """Generate a random int in range."""
        return self._rng.randint(min_val, max_val)
    
    def int_list(self, count: int, min_val: int = 0, max_val: int = 100) -> List[int]:
        """Generate a list of random ints."""
        return [self.int_value(min_val, max_val) for _ in range(count)]
    
    def bool_value(self) -> bool:
        """Generate a random boolean."""
        return self._rng.choice([True, False])
    
    def choice(self, items: List[Any]) -> Any:
        """Choose a random item from list."""
        return self._rng.choice(items)
    
    def sample(self, items: List[Any], k: int) -> List[Any]:
        """Sample k items from list without replacement."""
        return self._rng.sample(items, min(k, len(items)))


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def gen_goal_hit() -> DeterministicGenerator:
    """Generator with goal_hit seed."""
    return DeterministicGenerator(SEED_GOAL_HIT)


@pytest.fixture
def gen_sparse_density() -> DeterministicGenerator:
    """Generator with sparse_density seed."""
    return DeterministicGenerator(SEED_SPARSE_DENSITY)


@pytest.fixture
def gen_chain_length() -> DeterministicGenerator:
    """Generator with chain_length seed."""
    return DeterministicGenerator(SEED_CHAIN_LENGTH)


@pytest.fixture
def gen_multi_goal() -> DeterministicGenerator:
    """Generator with multi_goal seed."""
    return DeterministicGenerator(SEED_MULTI_GOAL)


@pytest.fixture
def gen_adapter() -> DeterministicGenerator:
    """Generator with adapter seed."""
    return DeterministicGenerator(SEED_ADAPTER)


@pytest.fixture
def gen_replay() -> DeterministicGenerator:
    """Generator with replay seed."""
    return DeterministicGenerator(SEED_REPLAY)


@pytest.fixture
def gen_base() -> DeterministicGenerator:
    """Generator with base seed for general use."""
    return DeterministicGenerator(SEED_BASE)


# ===========================================================================
# SLICE PARAMETER SETS FOR CROSS-SLICE TESTING
# ===========================================================================

SLICE_PARAMS = {
    "prop_depth4": {
        "min_success_rate": 0.95,
        "max_abstention_rate": 0.02,
        "min_throughput_uplift_pct": 5.0,
        "min_samples": 500,
    },
    "fol_eq_group": {
        "min_success_rate": 0.85,
        "max_abstention_rate": 0.10,
        "min_throughput_uplift_pct": 3.0,
        "min_samples": 300,
    },
    "fol_eq_ring": {
        "min_success_rate": 0.80,
        "max_abstention_rate": 0.15,
        "min_throughput_uplift_pct": 2.0,
        "min_samples": 300,
    },
    "linear_arith": {
        "min_success_rate": 0.70,
        "max_abstention_rate": 0.20,
        "min_throughput_uplift_pct": 0.0,
        "min_samples": 200,
    },
}


@pytest.fixture(params=list(SLICE_PARAMS.keys()))
def slice_id(request) -> str:
    """Parametrized fixture for slice IDs."""
    return request.param


@pytest.fixture
def slice_params(slice_id: str) -> Dict[str, Any]:
    """Get parameters for current slice."""
    return SLICE_PARAMS[slice_id]


# ===========================================================================
# PYTEST MARKERS
# ===========================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "phase2_metrics: marks tests as Phase II metrics battery"
    )
    config.addinivalue_line(
        "markers", "boundary: marks tests for boundary conditions"
    )
    config.addinivalue_line(
        "markers", "degenerate: marks tests for degenerate/edge cases"
    )
    config.addinivalue_line(
        "markers", "large_scale: marks tests with large data volumes"
    )
    config.addinivalue_line(
        "markers", "determinism: marks tests verifying deterministic behavior"
    )
    config.addinivalue_line(
        "markers", "schema: marks tests for schema validation"
    )
    config.addinivalue_line(
        "markers", "cross_slice: marks tests across multiple slice configs"
    )
    config.addinivalue_line(
        "markers", "replay: marks tests verifying replay equivalence"
    )
    config.addinivalue_line(
        "markers", "type_stability: marks tests verifying return types"
    )

