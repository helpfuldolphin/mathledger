# tests/phase2/metrics_adversarial/conftest.py
"""
Adversarial Metrics Test Suite - Shared Fixtures and Utilities

Provides:
- Fault injection generators
- Shadow metric implementations for mutation detection
- Equivalence oracle utilities
- High-volume batch generators
"""

import copy
import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import pytest


# ===========================================================================
# PRNG SEEDS - DO NOT CHANGE THESE VALUES
# ===========================================================================

SEED_ADVERSARIAL = 9001
SEED_MUTATION = 9002
SEED_ORACLE = 9003
SEED_REPLAY_EXTENDED = 9004
SEED_BATCH = 9005


# ===========================================================================
# FAULT INJECTION TYPES
# ===========================================================================

class FaultType(Enum):
    """Types of faults that can be injected."""
    MISSING_FIELD = "missing_field"
    WRONG_TYPE = "wrong_type"
    EXTREME_VALUE = "extreme_value"
    EMPTY_CONTAINER = "empty_container"
    NULL_VALUE = "null_value"
    NEGATIVE_VALUE = "negative_value"
    OVERFLOW_VALUE = "overflow_value"
    NAN_VALUE = "nan_value"
    INF_VALUE = "inf_value"


@dataclass
class InjectedFault:
    """Record of an injected fault for traceability."""
    fault_type: FaultType
    field_name: str
    original_value: Any
    injected_value: Any
    description: str


# ===========================================================================
# FAULT INJECTION GENERATOR
# ===========================================================================

class FaultInjector:
    """
    Deterministic fault injector for adversarial testing.
    
    Injects various fault types into metric function inputs
    while maintaining reproducibility via seeded PRNG.
    """
    
    # Extreme IEEE 754 values
    EXTREME_FLOATS = [
        1e308,           # Near max float
        -1e308,          # Near min float
        1e-308,          # Near min positive
        -1e-308,         # Near max negative subnormal
        float('inf'),    # Positive infinity
        float('-inf'),   # Negative infinity
        float('nan'),    # Not a number
        0.0,             # Zero
        -0.0,            # Negative zero
        1.7976931348623157e+308,  # Max float64
    ]
    
    EXTREME_INTS = [
        2**63 - 1,       # Max int64
        -(2**63),        # Min int64
        2**31 - 1,       # Max int32
        -(2**31),        # Min int32
        0,
        -1,
        1,
    ]
    
    WRONG_TYPE_SUBSTITUTES = {
        int: ["string", 3.14, None, [], {}, True],
        float: ["string", 42, None, [], {}, False],
        str: [42, 3.14, None, [], {}, True],
        set: ["string", 42, None, [], "not_a_set"],
        list: ["string", 42, None, {}, set()],
        dict: ["string", 42, None, [], set()],
    }
    
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._seed = seed
        self._faults_injected: List[InjectedFault] = []
    
    def reset(self) -> None:
        """Reset to initial seed state."""
        self._rng = random.Random(self._seed)
        self._faults_injected.clear()
    
    @property
    def faults(self) -> List[InjectedFault]:
        """Return list of injected faults."""
        return self._faults_injected.copy()
    
    def inject_missing_field(
        self,
        data: Dict[str, Any],
        field: str
    ) -> Tuple[Dict[str, Any], InjectedFault]:
        """Remove a field from a dictionary."""
        result = copy.deepcopy(data)
        original = result.pop(field, None)
        
        fault = InjectedFault(
            fault_type=FaultType.MISSING_FIELD,
            field_name=field,
            original_value=original,
            injected_value=None,
            description=f"Removed field '{field}'"
        )
        self._faults_injected.append(fault)
        return result, fault
    
    def inject_wrong_type(
        self,
        data: Dict[str, Any],
        field: str,
        expected_type: type
    ) -> Tuple[Dict[str, Any], InjectedFault]:
        """Replace a field with a value of the wrong type."""
        result = copy.deepcopy(data)
        original = result.get(field)
        
        substitutes = self.WRONG_TYPE_SUBSTITUTES.get(expected_type, ["invalid"])
        wrong_value = self._rng.choice(substitutes)
        result[field] = wrong_value
        
        fault = InjectedFault(
            fault_type=FaultType.WRONG_TYPE,
            field_name=field,
            original_value=original,
            injected_value=wrong_value,
            description=f"Replaced '{field}' with {type(wrong_value).__name__}"
        )
        self._faults_injected.append(fault)
        return result, fault
    
    def inject_extreme_float(
        self,
        data: Dict[str, Any],
        field: str
    ) -> Tuple[Dict[str, Any], InjectedFault]:
        """Replace a field with an extreme float value."""
        result = copy.deepcopy(data)
        original = result.get(field)
        
        extreme = self._rng.choice(self.EXTREME_FLOATS)
        result[field] = extreme
        
        fault = InjectedFault(
            fault_type=FaultType.EXTREME_VALUE,
            field_name=field,
            original_value=original,
            injected_value=extreme,
            description=f"Injected extreme float {extreme} into '{field}'"
        )
        self._faults_injected.append(fault)
        return result, fault
    
    def inject_extreme_int(
        self,
        data: Dict[str, Any],
        field: str
    ) -> Tuple[Dict[str, Any], InjectedFault]:
        """Replace a field with an extreme integer value."""
        result = copy.deepcopy(data)
        original = result.get(field)
        
        extreme = self._rng.choice(self.EXTREME_INTS)
        result[field] = extreme
        
        fault = InjectedFault(
            fault_type=FaultType.EXTREME_VALUE,
            field_name=field,
            original_value=original,
            injected_value=extreme,
            description=f"Injected extreme int {extreme} into '{field}'"
        )
        self._faults_injected.append(fault)
        return result, fault
    
    def inject_empty_container(
        self,
        data: Dict[str, Any],
        field: str,
        container_type: type
    ) -> Tuple[Dict[str, Any], InjectedFault]:
        """Replace a field with an empty container."""
        result = copy.deepcopy(data)
        original = result.get(field)
        
        empty_containers = {
            list: [],
            dict: {},
            set: set(),
            str: "",
            tuple: (),
        }
        empty = empty_containers.get(container_type, None)
        result[field] = empty
        
        fault = InjectedFault(
            fault_type=FaultType.EMPTY_CONTAINER,
            field_name=field,
            original_value=original,
            injected_value=empty,
            description=f"Injected empty {container_type.__name__} into '{field}'"
        )
        self._faults_injected.append(fault)
        return result, fault
    
    def inject_null(
        self,
        data: Dict[str, Any],
        field: str
    ) -> Tuple[Dict[str, Any], InjectedFault]:
        """Replace a field with None."""
        result = copy.deepcopy(data)
        original = result.get(field)
        result[field] = None
        
        fault = InjectedFault(
            fault_type=FaultType.NULL_VALUE,
            field_name=field,
            original_value=original,
            injected_value=None,
            description=f"Injected None into '{field}'"
        )
        self._faults_injected.append(fault)
        return result, fault
    
    def inject_nan(
        self,
        data: Dict[str, Any],
        field: str
    ) -> Tuple[Dict[str, Any], InjectedFault]:
        """Replace a numeric field with NaN."""
        result = copy.deepcopy(data)
        original = result.get(field)
        result[field] = float('nan')
        
        fault = InjectedFault(
            fault_type=FaultType.NAN_VALUE,
            field_name=field,
            original_value=original,
            injected_value=float('nan'),
            description=f"Injected NaN into '{field}'"
        )
        self._faults_injected.append(fault)
        return result, fault
    
    def inject_inf(
        self,
        data: Dict[str, Any],
        field: str,
        negative: bool = False
    ) -> Tuple[Dict[str, Any], InjectedFault]:
        """Replace a numeric field with infinity."""
        result = copy.deepcopy(data)
        original = result.get(field)
        inf_val = float('-inf') if negative else float('inf')
        result[field] = inf_val
        
        fault = InjectedFault(
            fault_type=FaultType.INF_VALUE,
            field_name=field,
            original_value=original,
            injected_value=inf_val,
            description=f"Injected {inf_val} into '{field}'"
        )
        self._faults_injected.append(fault)
        return result, fault
    
    def random_fault(
        self,
        data: Dict[str, Any],
        field: str,
        expected_type: type
    ) -> Tuple[Dict[str, Any], InjectedFault]:
        """Inject a random fault type."""
        fault_methods = [
            lambda: self.inject_missing_field(data, field),
            lambda: self.inject_wrong_type(data, field, expected_type),
            lambda: self.inject_null(data, field),
        ]
        
        if expected_type in (int, float):
            fault_methods.extend([
                lambda: self.inject_extreme_float(data, field),
                lambda: self.inject_extreme_int(data, field),
                lambda: self.inject_nan(data, field),
                lambda: self.inject_inf(data, field),
            ])
        
        if expected_type in (list, dict, set, str):
            fault_methods.append(
                lambda: self.inject_empty_container(data, field, expected_type)
            )
        
        method = self._rng.choice(fault_methods)
        return method()


# ===========================================================================
# SHADOW METRIC IMPLEMENTATIONS (For Mutation Detection)
# ===========================================================================

class ShadowMetrics:
    """
    Shadow implementations of metric functions for mutation detection.
    
    These implementations are mathematically equivalent to production
    but use different algorithms to detect implementation bugs.
    """
    
    @staticmethod
    def compute_goal_hit_shadow(
        verified_statements: List[Dict[str, Any]],
        target_hashes: Set[str],
        min_total_verified: int,
    ) -> Tuple[bool, float]:
        """
        Shadow implementation of compute_goal_hit.
        
        Uses explicit loop instead of set intersection for verification.
        """
        hits = 0
        seen_hashes: Set[str] = set()
        
        for stmt in verified_statements:
            h = stmt.get('hash')
            if h is not None and h not in seen_hashes:
                seen_hashes.add(h)
                if h in target_hashes:
                    hits += 1
        
        success = hits >= min_total_verified
        return success, float(hits)
    
    @staticmethod
    def compute_sparse_success_shadow(
        verified_count: int,
        attempted_count: int,
        min_verified: int,
    ) -> Tuple[bool, float]:
        """
        Shadow implementation of compute_sparse_success.
        
        Identical logic (trivial function) but explicit comparison.
        """
        _ = attempted_count  # Explicitly ignored
        is_success = True if verified_count >= min_verified else False
        return is_success, float(verified_count)
    
    @staticmethod
    def compute_chain_success_shadow(
        verified_statements: List[Dict[str, Any]],
        dependency_graph: Dict[str, List[str]],
        chain_target_hash: str,
        min_chain_length: int,
    ) -> Tuple[bool, float]:
        """
        Shadow implementation of compute_chain_success.
        
        Uses iterative BFS instead of recursive DFS for chain length.
        """
        verified_set = {s['hash'] for s in verified_statements}
        
        if chain_target_hash not in verified_set:
            return (0 >= min_chain_length), 0.0
        
        # Iterative approach with queue
        # For each node, track the longest path to it
        longest_path: Dict[str, int] = {}
        
        # Topological-ish processing
        # Start from nodes with no dependencies
        all_nodes = set(dependency_graph.keys()) | {chain_target_hash}
        for stmt in verified_statements:
            all_nodes.add(stmt['hash'])
        
        # Initialize
        for node in all_nodes:
            if node in verified_set:
                deps = dependency_graph.get(node, [])
                if not deps:
                    longest_path[node] = 1
        
        # Iterate until stable
        changed = True
        max_iterations = len(all_nodes) + 10
        iterations = 0
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for node in all_nodes:
                if node not in verified_set:
                    continue
                
                deps = dependency_graph.get(node, [])
                if not deps:
                    if node not in longest_path:
                        longest_path[node] = 1
                        changed = True
                else:
                    max_dep_len = 0
                    all_deps_computed = True
                    
                    for dep in deps:
                        if dep in verified_set:
                            if dep in longest_path:
                                max_dep_len = max(max_dep_len, longest_path[dep])
                            else:
                                all_deps_computed = False
                    
                    new_len = 1 + max_dep_len
                    if node not in longest_path or longest_path[node] < new_len:
                        longest_path[node] = new_len
                        changed = True
        
        chain_len = longest_path.get(chain_target_hash, 0)
        success = chain_len >= min_chain_length
        return success, float(chain_len)
    
    @staticmethod
    def compute_multi_goal_success_shadow(
        verified_hashes: Set[str],
        required_goal_hashes: Set[str],
    ) -> Tuple[bool, float]:
        """
        Shadow implementation of compute_multi_goal_success.
        
        Uses explicit iteration instead of set intersection.
        """
        if not required_goal_hashes:
            return True, 0.0
        
        met_count = 0
        for goal in required_goal_hashes:
            if goal in verified_hashes:
                met_count += 1
        
        success = (met_count == len(required_goal_hashes))
        return success, float(met_count)


# ===========================================================================
# MUTATION OPERATORS
# ===========================================================================

class MutationOperator:
    """
    Applies systematic mutations to metric inputs for differential testing.
    """
    
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._seed = seed
    
    def reset(self) -> None:
        self._rng = random.Random(self._seed)
    
    def mutate_int(self, value: int, delta_range: int = 2) -> List[int]:
        """Generate mutations of an integer value."""
        mutations = [
            value + 1,
            value - 1,
            value + 2,
            value - 2,
            0,
            -value if value != 0 else 1,
        ]
        if value > 0:
            mutations.append(value * 2)
        return list(set(mutations))
    
    def mutate_set(self, s: Set[str]) -> List[Set[str]]:
        """Generate mutations of a set."""
        mutations = []
        s_list = list(s)
        
        # Remove one element
        if s_list:
            for i in range(min(3, len(s_list))):
                idx = self._rng.randint(0, len(s_list) - 1)
                mutations.append(s - {s_list[idx]})
        
        # Add one element
        mutations.append(s | {f"mutated_{self._rng.randint(0, 999)}"})
        
        # Empty set
        mutations.append(set())
        
        # Shuffle/permute (should be equivalent)
        if len(s) > 1:
            shuffled = list(s)
            self._rng.shuffle(shuffled)
            mutations.append(set(shuffled))
        
        return mutations
    
    def mutate_statements(
        self,
        statements: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Generate mutations of a statement list."""
        mutations = []
        
        # Remove one statement
        if statements:
            for i in range(min(3, len(statements))):
                idx = self._rng.randint(0, len(statements) - 1)
                mutations.append(statements[:idx] + statements[idx+1:])
        
        # Add duplicate
        if statements:
            mutations.append(statements + [statements[0]])
        
        # Shuffle order
        if len(statements) > 1:
            shuffled = statements.copy()
            self._rng.shuffle(shuffled)
            mutations.append(shuffled)
        
        # Empty list
        mutations.append([])
        
        return mutations


# ===========================================================================
# EQUIVALENCE ORACLE
# ===========================================================================

class EquivalenceOracle:
    """
    Verifies that mathematically equivalent inputs produce identical outputs.
    """
    
    @staticmethod
    def is_equivalent_goal_hit(
        result1: Tuple[bool, float],
        result2: Tuple[bool, float]
    ) -> bool:
        """Check if two goal_hit results are equivalent."""
        return result1[0] == result2[0] and result1[1] == result2[1]
    
    @staticmethod
    def is_equivalent_float(a: float, b: float, epsilon: float = 1e-9) -> bool:
        """Check if two floats are equivalent within epsilon."""
        if math.isnan(a) and math.isnan(b):
            return True
        if math.isinf(a) and math.isinf(b):
            return (a > 0) == (b > 0)
        return abs(a - b) < epsilon
    
    @staticmethod
    def generate_permutations(
        statements: List[Dict[str, Any]],
        max_permutations: int = 10
    ) -> List[List[Dict[str, Any]]]:
        """Generate permutations of statement order."""
        import itertools
        
        if len(statements) <= 1:
            return [statements]
        
        if len(statements) <= 5:
            # Generate all permutations for small lists
            return [list(p) for p in itertools.permutations(statements)][:max_permutations]
        else:
            # Sample random permutations for large lists
            rng = random.Random(SEED_ORACLE)
            perms = []
            for _ in range(max_permutations):
                shuffled = statements.copy()
                rng.shuffle(shuffled)
                perms.append(shuffled)
            return perms


# ===========================================================================
# BATCH GENERATOR FOR HIGH-VOLUME TESTS
# ===========================================================================

class BatchGenerator:
    """
    Generates large batches of test data for performance testing.
    """
    
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._seed = seed
    
    def reset(self) -> None:
        self._rng = random.Random(self._seed)
    
    def generate_goal_hit_batch(
        self,
        batch_size: int,
        max_statements: int = 50,
        max_targets: int = 20
    ) -> List[Tuple[List[Dict[str, Any]], Set[str], int]]:
        """Generate a batch of goal_hit inputs."""
        batch = []
        for _ in range(batch_size):
            num_statements = self._rng.randint(1, max_statements)
            num_targets = self._rng.randint(1, max_targets)
            
            statements = [{"hash": f"h{self._rng.randint(0, 999)}"} for _ in range(num_statements)]
            targets = {f"h{self._rng.randint(0, 999)}" for _ in range(num_targets)}
            min_hits = self._rng.randint(0, min(num_targets, num_statements))
            
            batch.append((statements, targets, min_hits))
        
        return batch
    
    def generate_sparse_success_batch(
        self,
        batch_size: int,
        max_verified: int = 1000,
        max_attempted: int = 2000
    ) -> List[Tuple[int, int, int]]:
        """Generate a batch of sparse_success inputs."""
        batch = []
        for _ in range(batch_size):
            verified = self._rng.randint(0, max_verified)
            attempted = self._rng.randint(verified, max_attempted)
            min_ver = self._rng.randint(0, max_verified)
            batch.append((verified, attempted, min_ver))
        
        return batch
    
    def generate_multi_goal_batch(
        self,
        batch_size: int,
        max_verified: int = 100,
        max_required: int = 50
    ) -> List[Tuple[Set[str], Set[str]]]:
        """Generate a batch of multi_goal inputs."""
        batch = []
        for _ in range(batch_size):
            num_verified = self._rng.randint(0, max_verified)
            num_required = self._rng.randint(0, max_required)
            
            verified = {f"h{self._rng.randint(0, 999)}" for _ in range(num_verified)}
            required = {f"h{self._rng.randint(0, 999)}" for _ in range(num_required)}
            
            batch.append((verified, required))
        
        return batch


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def fault_injector() -> FaultInjector:
    """Create a fault injector with adversarial seed."""
    return FaultInjector(SEED_ADVERSARIAL)


@pytest.fixture
def shadow_metrics() -> ShadowMetrics:
    """Create shadow metrics instance."""
    return ShadowMetrics()


@pytest.fixture
def mutation_operator() -> MutationOperator:
    """Create mutation operator."""
    return MutationOperator(SEED_MUTATION)


@pytest.fixture
def equivalence_oracle() -> EquivalenceOracle:
    """Create equivalence oracle."""
    return EquivalenceOracle()


@pytest.fixture
def batch_generator() -> BatchGenerator:
    """Create batch generator."""
    return BatchGenerator(SEED_BATCH)


# ===========================================================================
# PYTEST MARKERS
# ===========================================================================

def pytest_configure(config):
    """Register adversarial test markers."""
    config.addinivalue_line(
        "markers", "adversarial: marks tests as adversarial fault injection"
    )
    config.addinivalue_line(
        "markers", "mutation: marks tests for mutation detection"
    )
    config.addinivalue_line(
        "markers", "oracle: marks tests using equivalence oracle"
    )
    config.addinivalue_line(
        "markers", "high_volume: marks tests with large batch sizes"
    )
    config.addinivalue_line(
        "markers", "entropy: marks tests checking for entropy leaks"
    )

