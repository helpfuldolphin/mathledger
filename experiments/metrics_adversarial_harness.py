#!/usr/bin/env python
"""
Metric Torture Test — Adversarial Harness CLI

Institutional Standard for Metric Robustness Testing

A one-button CLI for running adversarial metric tests under fault injection,
mutation, and replay conditions.

Usage:
    python -m experiments.metrics_adversarial_harness --mode fault --metric-kind goal_hit
    python -m experiments.metrics_adversarial_harness --profile standard --all-metrics
    python -m experiments.metrics_adversarial_harness --coverage
    python -m experiments.metrics_adversarial_harness --regression-radar

Profiles (Contracts):
    fast     - ~50 tests per metric, modes: fault+replay (quick CI check)
    standard - ~250 tests per metric, modes: fault+mutation+replay (default)
    full     - ~1000+ tests per metric, modes: fault+mutation+replay (nightly)

Modes:
    fault     - Apply fault injections (missing fields, wrong types, extreme values)
    mutation  - Apply parameter mutations (±1, ±2, boundary crossings)
    replay    - Verify determinism via replay (run twice, compare results)
    all       - Run all modes sequentially

Special Modes:
    --coverage         - Output JSON coverage report (CI contract)
    --regression-radar - Run fixed battery, compare against expected outputs

NO METRIC INTERPRETATION: This harness verifies mechanical correctness only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

# Import from canonical backend implementation
from backend.substrate.slice_success_metrics import (
    compute_goal_hit,
    compute_sparse_success,
    compute_chain_success,
    compute_multi_goal_success,
)


# ===========================================================================
# CONSTANTS
# ===========================================================================

SEED_HARNESS = 7777
METRIC_KINDS = ["goal_hit", "density", "chain_length", "multi_goal"]
MODES = ["fault", "mutation", "replay", "all"]
PROFILES = ["fast", "standard", "full"]


# ===========================================================================
# PROFILE CONTRACTS — Institutional Standard
# ===========================================================================

@dataclass(frozen=True)
class ProfileContract:
    """
    Immutable contract specifying minimum guarantees for a profile.
    
    These are institutional requirements — profiles MUST meet these minimums.
    """
    name: str
    min_tests_per_metric: int
    modes: Tuple[str, ...]
    min_fault_types: int
    min_mutation_categories: int
    description: str


# Profile contracts as immutable specifications
PROFILE_CONTRACTS: Dict[str, ProfileContract] = {
    "fast": ProfileContract(
        name="fast",
        min_tests_per_metric=50,
        modes=("fault", "replay"),
        min_fault_types=3,
        min_mutation_categories=0,  # No mutation in fast
        description="Quick CI check (~50 tests/metric)",
    ),
    "standard": ProfileContract(
        name="standard",
        min_tests_per_metric=250,
        modes=("fault", "mutation", "replay"),
        min_fault_types=5,
        min_mutation_categories=2,
        description="Standard coverage (~250 tests/metric)",
    ),
    "full": ProfileContract(
        name="full",
        min_tests_per_metric=1000,
        modes=("fault", "mutation", "replay"),
        min_fault_types=8,
        min_mutation_categories=4,
        description="Full nightly coverage (~1000+ tests/metric)",
    ),
}

# Runtime configuration (must meet contracts)
PROFILE_CONFIG = {
    "fast": {
        "n_tests": 50,
        "modes": ["fault", "replay"],
        "description": "Quick CI check (~50 tests/metric)",
    },
    "standard": {
        "n_tests": 250,
        "modes": ["fault", "mutation", "replay"],
        "description": "Standard coverage (~250 tests/metric)",
    },
    "full": {
        "n_tests": 1000,
        "modes": ["fault", "mutation", "replay"],
        "description": "Full nightly coverage (~1000+ tests/metric)",
    },
}


def get_profile_contract(profile: str) -> ProfileContract:
    """Get the contract for a profile."""
    return PROFILE_CONTRACTS[profile]


def validate_profile_meets_contract(profile: str) -> Tuple[bool, List[str]]:
    """
    Validate that a profile configuration meets its contract.
    
    Returns: (passes, list of violations)
    """
    contract = PROFILE_CONTRACTS[profile]
    config = PROFILE_CONFIG[profile]
    
    violations = []
    
    # Check min tests
    if config["n_tests"] < contract.min_tests_per_metric:
        violations.append(
            f"Tests {config['n_tests']} < min {contract.min_tests_per_metric}"
        )
    
    # Check modes
    config_modes = set(config["modes"])
    contract_modes = set(contract.modes)
    missing_modes = contract_modes - config_modes
    if missing_modes:
        violations.append(f"Missing required modes: {missing_modes}")
    
    return len(violations) == 0, violations


def get_profile_contract_json() -> str:
    """Export profile contracts as JSON for CI integration."""
    contracts = {}
    for name, contract in PROFILE_CONTRACTS.items():
        contracts[name] = {
            "min_tests_per_metric": contract.min_tests_per_metric,
            "modes": list(contract.modes),
            "min_fault_types": contract.min_fault_types,
            "min_mutation_categories": contract.min_mutation_categories,
            "description": contract.description,
        }
    return json.dumps(contracts, indent=2)


# ===========================================================================
# SCENARIO REGISTRY — Named Adversarial Profiles
# ===========================================================================

@dataclass(frozen=True)
class Scenario:
    """
    A named adversarial scenario for coverage by name.
    
    Scenarios provide declarative, named test configurations that
    map to profile contracts and specific metric subsets.
    """
    name: str
    profile: str  # fast|standard|full
    metric_kinds: Tuple[str, ...]
    modes: Tuple[str, ...]
    description: str


# Scenario registry — deterministic, sorted by name
SCENARIOS: Dict[str, Scenario] = {
    "baseline_sanity": Scenario(
        name="baseline_sanity",
        profile="fast",
        metric_kinds=tuple(METRIC_KINDS),
        modes=("fault", "replay"),
        description="Quick sanity check across all metrics",
    ),
    "goal_hit_boundary": Scenario(
        name="goal_hit_boundary",
        profile="standard",
        metric_kinds=("goal_hit",),
        modes=("fault", "mutation", "replay"),
        description="Boundary condition focus on goal_hit metric",
    ),
    "density_stress": Scenario(
        name="density_stress",
        profile="full",
        metric_kinds=("density",),
        modes=("fault", "mutation", "replay"),
        description="High-volume stress test for density metric",
    ),
    "chain_length_deep": Scenario(
        name="chain_length_deep",
        profile="standard",
        metric_kinds=("chain_length",),
        modes=("fault", "mutation", "replay"),
        description="Deep chain verification scenarios",
    ),
    "multi_goal_coverage": Scenario(
        name="multi_goal_coverage",
        profile="standard",
        metric_kinds=("multi_goal",),
        modes=("fault", "mutation", "replay"),
        description="Multi-goal success criteria coverage",
    ),
    "multi_metric_stress": Scenario(
        name="multi_metric_stress",
        profile="full",
        metric_kinds=tuple(METRIC_KINDS),
        modes=("fault", "mutation", "replay"),
        description="Full stress test across all metrics (nightly)",
    ),
    "ci_quick": Scenario(
        name="ci_quick",
        profile="fast",
        metric_kinds=("goal_hit", "density"),
        modes=("fault", "replay"),
        description="Quick CI gate for critical metrics",
    ),
    "mutation_focus": Scenario(
        name="mutation_focus",
        profile="standard",
        metric_kinds=tuple(METRIC_KINDS),
        modes=("mutation",),
        description="Mutation-only testing across all metrics",
    ),
}


def list_scenarios(filter_profile: Optional[str] = None) -> List[Scenario]:
    """
    List available scenarios, optionally filtered by profile.
    
    Args:
        filter_profile: Optional profile name to filter by (fast|standard|full)
    
    Returns:
        List of Scenario objects, sorted by name for determinism.
    """
    scenarios = list(SCENARIOS.values())
    
    if filter_profile is not None:
        scenarios = [s for s in scenarios if s.profile == filter_profile]
    
    # Sort by name for determinism
    return sorted(scenarios, key=lambda s: s.name)


def get_scenario(name: str) -> Scenario:
    """Get a scenario by name."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]


def validate_scenario(scenario: Scenario) -> Tuple[bool, List[str]]:
    """
    Validate that a scenario references a valid ProfileContract.
    
    Returns: (valid, list of violations)
    """
    violations = []
    
    # Check profile exists
    if scenario.profile not in PROFILE_CONTRACTS:
        violations.append(f"Unknown profile: {scenario.profile}")
        return False, violations
    
    contract = PROFILE_CONTRACTS[scenario.profile]
    
    # Check modes are subset of contract modes
    for mode in scenario.modes:
        if mode not in contract.modes and mode != "all":
            violations.append(f"Mode '{mode}' not in profile '{scenario.profile}'")
    
    # Check metric kinds are valid
    for mk in scenario.metric_kinds:
        if mk not in METRIC_KINDS:
            violations.append(f"Unknown metric kind: {mk}")
    
    return len(violations) == 0, violations


def list_scenarios_json() -> str:
    """Export scenario registry as JSON."""
    scenarios = {}
    for name, scenario in sorted(SCENARIOS.items()):
        scenarios[name] = {
            "profile": scenario.profile,
            "metric_kinds": list(scenario.metric_kinds),
            "modes": list(scenario.modes),
            "description": scenario.description,
        }
    return json.dumps(scenarios, indent=2, sort_keys=True)


# ===========================================================================
# COVERAGE REPORT CONTRACT — CI-Readable Format
# ===========================================================================

# Required keys in coverage report (CI contract)
COVERAGE_REPORT_REQUIRED_KEYS = [
    "metric_kinds",
    "fault_types_per_metric",
    "mutation_categories_per_metric",
    "replay_sizes_per_metric",
    "total_fault_types",
    "total_metrics",
]


# ===========================================================================
# FAULT TYPES
# ===========================================================================

class FaultType(Enum):
    MISSING_FIELD = "missing_field"
    WRONG_TYPE = "wrong_type"
    EXTREME_VALUE = "extreme_value"
    EMPTY_CONTAINER = "empty_container"
    NULL_VALUE = "null_value"
    NAN_VALUE = "nan_value"
    INF_VALUE = "inf_value"
    NEGATIVE_VALUE = "negative_value"


# All fault type names for coverage reporting
ALL_FAULT_TYPES = [ft.value for ft in FaultType]

# Mutation categories
MUTATION_CATEGORIES = {
    "goal_hit": ["threshold_plus_1", "threshold_plus_2", "threshold_minus_1", "threshold_minus_2"],
    "density": ["min_verified_plus_1", "min_verified_plus_2", "min_verified_minus_1", "min_verified_minus_2"],
    "chain_length": ["min_length_plus_1", "min_length_plus_2", "min_length_minus_1", "min_length_minus_2"],
    "multi_goal": ["required_add_element", "required_remove_element"],
}


# ===========================================================================
# DATA CLASSES
# ===========================================================================

@dataclass
class TestCase:
    """A single test case with inputs and expected behavior."""
    metric_kind: str
    inputs: Dict[str, Any]
    fault_type: Optional[FaultType] = None
    fault_description: str = ""


@dataclass
class TestResult:
    """Result of a single test case execution."""
    test_case: TestCase
    success: bool
    production_result: Optional[Tuple[bool, float]] = None
    shadow_result: Optional[Tuple[bool, float]] = None
    error: Optional[str] = None
    mismatch: bool = False
    mismatch_type: str = ""


@dataclass
class HarnessSummary:
    """Summary of harness execution."""
    mode: str
    metric_kind: str
    total_cases: int
    passed: int
    failed: int
    mismatches: int
    errors: int
    fault_divergences: Dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0


@dataclass
class CoverageReport:
    """
    Coverage report for CI decoration.
    
    Required keys (CI contract):
    - metric_kinds: List of all metric kinds covered
    - fault_types_per_metric: Dict mapping metric → fault types
    - mutation_categories_per_metric: Dict mapping metric → mutation categories
    - replay_sizes_per_metric: Dict mapping metric → replay test count
    - total_fault_types: Total unique fault types
    - total_metrics: Total metrics covered
    """
    metric_kinds: List[str]
    fault_types_per_metric: Dict[str, List[str]]
    mutation_categories_per_metric: Dict[str, List[str]]
    replay_sizes_per_metric: Dict[str, int]
    total_fault_types: int
    total_metrics: int


@dataclass
class RegressionResult:
    """Result of regression radar check."""
    metric_kind: str
    test_name: str
    expected_hash: str
    actual_hash: str
    match: bool


# ===========================================================================
# INPUT GENERATORS
# ===========================================================================

class InputGenerator:
    """Generates valid inputs for metric functions."""
    
    def __init__(self, seed: int = SEED_HARNESS) -> None:
        self._rng = random.Random(seed)
        self._seed = seed
    
    def reset(self) -> None:
        self._rng = random.Random(self._seed)
    
    def generate_goal_hit_inputs(self) -> Dict[str, Any]:
        """Generate valid goal_hit inputs."""
        num_statements = self._rng.randint(5, 50)
        num_targets = self._rng.randint(2, 15)
        
        statements = [{"hash": f"h{self._rng.randint(0, 999)}"} for _ in range(num_statements)]
        targets = {f"h{self._rng.randint(0, 999)}" for _ in range(num_targets)}
        min_hits = self._rng.randint(0, min(5, num_targets))
        
        return {
            "statements": statements,
            "targets": targets,
            "min_hits": min_hits,
        }
    
    def generate_density_inputs(self) -> Dict[str, Any]:
        """Generate valid density (sparse_success) inputs."""
        verified = self._rng.randint(0, 500)
        attempted = self._rng.randint(verified, 1000)
        min_verified = self._rng.randint(0, 250)
        
        return {
            "verified_count": verified,
            "attempted_count": attempted,
            "min_verified": min_verified,
        }
    
    def generate_chain_length_inputs(self) -> Dict[str, Any]:
        """Generate valid chain_length inputs."""
        depth = self._rng.randint(3, 15)
        hashes = [f"h{i}" for i in range(depth)]
        
        graph: Dict[str, List[str]] = {}
        for i in range(1, depth):
            if self._rng.random() < 0.8:
                graph[hashes[i]] = [hashes[i - 1]]
            else:
                branch_to = self._rng.randint(0, i - 1)
                graph[hashes[i]] = [hashes[branch_to]]
        
        statements = [{"hash": h} for h in hashes]
        target = hashes[-1]
        min_length = self._rng.randint(1, depth)
        
        return {
            "statements": statements,
            "graph": graph,
            "target": target,
            "min_length": min_length,
        }
    
    def generate_multi_goal_inputs(self) -> Dict[str, Any]:
        """Generate valid multi_goal inputs."""
        num_verified = self._rng.randint(10, 100)
        num_required = self._rng.randint(2, 20)
        
        verified = {f"h{self._rng.randint(0, 999)}" for _ in range(num_verified)}
        required = {f"h{self._rng.randint(0, 999)}" for _ in range(num_required)}
        
        return {
            "verified": verified,
            "required": required,
        }
    
    def generate_inputs(self, metric_kind: str) -> Dict[str, Any]:
        """Generate inputs for the specified metric kind."""
        generators = {
            "goal_hit": self.generate_goal_hit_inputs,
            "density": self.generate_density_inputs,
            "chain_length": self.generate_chain_length_inputs,
            "multi_goal": self.generate_multi_goal_inputs,
        }
        return generators[metric_kind]()


# ===========================================================================
# FAULT INJECTOR
# ===========================================================================

class FaultInjector:
    """Injects faults into metric inputs."""
    
    EXTREME_FLOATS = [1e308, -1e308, float('inf'), float('-inf'), float('nan')]
    EXTREME_INTS = [2**63 - 1, -(2**63), 2**31 - 1, -(2**31)]
    
    def __init__(self, seed: int = SEED_HARNESS) -> None:
        self._rng = random.Random(seed)
    
    def get_applicable_faults(self, metric_kind: str) -> List[FaultType]:
        """Get fault types applicable to a metric kind."""
        applicable = [
            FaultType.EMPTY_CONTAINER,
            FaultType.EXTREME_VALUE,
            FaultType.NEGATIVE_VALUE,
        ]
        
        if metric_kind in ("goal_hit", "chain_length"):
            applicable.extend([FaultType.MISSING_FIELD, FaultType.WRONG_TYPE, FaultType.NULL_VALUE])
        
        if metric_kind == "density":
            applicable.extend([FaultType.WRONG_TYPE, FaultType.NAN_VALUE, FaultType.INF_VALUE])
        
        if metric_kind == "multi_goal":
            applicable.append(FaultType.NULL_VALUE)
        
        return applicable
    
    def inject_fault(
        self,
        inputs: Dict[str, Any],
        metric_kind: str,
        fault_type: FaultType
    ) -> Tuple[Dict[str, Any], str]:
        """Inject a fault into the inputs. Returns (faulted_inputs, description)."""
        faulted = copy.deepcopy(inputs)
        description = ""
        
        if fault_type == FaultType.MISSING_FIELD:
            faulted, description = self._inject_missing_field(faulted, metric_kind)
        elif fault_type == FaultType.WRONG_TYPE:
            faulted, description = self._inject_wrong_type(faulted, metric_kind)
        elif fault_type == FaultType.EXTREME_VALUE:
            faulted, description = self._inject_extreme_value(faulted, metric_kind)
        elif fault_type == FaultType.EMPTY_CONTAINER:
            faulted, description = self._inject_empty_container(faulted, metric_kind)
        elif fault_type == FaultType.NULL_VALUE:
            faulted, description = self._inject_null(faulted, metric_kind)
        elif fault_type == FaultType.NAN_VALUE:
            faulted, description = self._inject_nan(faulted, metric_kind)
        elif fault_type == FaultType.INF_VALUE:
            faulted, description = self._inject_inf(faulted, metric_kind)
        elif fault_type == FaultType.NEGATIVE_VALUE:
            faulted, description = self._inject_negative(faulted, metric_kind)
        
        return faulted, description
    
    def _inject_missing_field(self, inputs: Dict[str, Any], metric_kind: str) -> Tuple[Dict[str, Any], str]:
        if metric_kind == "goal_hit" and inputs.get("statements"):
            idx = self._rng.randint(0, len(inputs["statements"]) - 1)
            del inputs["statements"][idx]["hash"]
            return inputs, f"Removed 'hash' from statement {idx}"
        elif metric_kind == "chain_length" and inputs.get("statements"):
            idx = self._rng.randint(0, len(inputs["statements"]) - 1)
            del inputs["statements"][idx]["hash"]
            return inputs, f"Removed 'hash' from statement {idx}"
        return inputs, "No field removed"
    
    def _inject_wrong_type(self, inputs: Dict[str, Any], metric_kind: str) -> Tuple[Dict[str, Any], str]:
        if metric_kind == "density":
            inputs["verified_count"] = "not_an_int"
            return inputs, "Set verified_count to string"
        elif metric_kind == "goal_hit":
            inputs["statements"] = "not_a_list"
            return inputs, "Set statements to string"
        return inputs, "No type changed"
    
    def _inject_extreme_value(self, inputs: Dict[str, Any], metric_kind: str) -> Tuple[Dict[str, Any], str]:
        if metric_kind == "density":
            extreme = self._rng.choice(self.EXTREME_INTS)
            inputs["verified_count"] = extreme
            return inputs, f"Set verified_count to {extreme}"
        elif metric_kind in ("goal_hit", "multi_goal"):
            extreme = self._rng.choice(self.EXTREME_INTS)
            inputs["min_hits"] = extreme
            return inputs, f"Set min_hits to {extreme}"
        return inputs, "No extreme value injected"
    
    def _inject_empty_container(self, inputs: Dict[str, Any], metric_kind: str) -> Tuple[Dict[str, Any], str]:
        if metric_kind == "goal_hit":
            inputs["statements"] = []
            return inputs, "Set statements to empty list"
        elif metric_kind == "multi_goal":
            inputs["verified"] = set()
            return inputs, "Set verified to empty set"
        elif metric_kind == "chain_length":
            inputs["graph"] = {}
            return inputs, "Set graph to empty dict"
        return inputs, "No container emptied"
    
    def _inject_null(self, inputs: Dict[str, Any], metric_kind: str) -> Tuple[Dict[str, Any], str]:
        if metric_kind == "goal_hit" and inputs.get("statements"):
            idx = self._rng.randint(0, len(inputs["statements"]) - 1)
            inputs["statements"][idx]["hash"] = None
            return inputs, f"Set statement {idx} hash to None"
        elif metric_kind == "chain_length":
            inputs["target"] = None
            return inputs, "Set target to None"
        return inputs, "No null injected"
    
    def _inject_nan(self, inputs: Dict[str, Any], metric_kind: str) -> Tuple[Dict[str, Any], str]:
        if metric_kind == "density":
            inputs["verified_count"] = float('nan')
            return inputs, "Set verified_count to NaN"
        return inputs, "No NaN injected"
    
    def _inject_inf(self, inputs: Dict[str, Any], metric_kind: str) -> Tuple[Dict[str, Any], str]:
        if metric_kind == "density":
            inputs["verified_count"] = float('inf')
            return inputs, "Set verified_count to inf"
        return inputs, "No inf injected"
    
    def _inject_negative(self, inputs: Dict[str, Any], metric_kind: str) -> Tuple[Dict[str, Any], str]:
        if metric_kind == "density":
            inputs["verified_count"] = -100
            return inputs, "Set verified_count to -100"
        elif metric_kind in ("goal_hit", "chain_length"):
            inputs["min_hits"] = -5 if "min_hits" in inputs else inputs
            inputs["min_length"] = -5 if "min_length" in inputs else inputs
            return inputs, "Set threshold to negative"
        return inputs, "No negative injected"


# ===========================================================================
# MUTATION OPERATOR
# ===========================================================================

class MutationOperator:
    """Applies systematic mutations to inputs."""
    
    def __init__(self, seed: int = SEED_HARNESS) -> None:
        self._rng = random.Random(seed)
    
    def mutate_boundary(
        self,
        inputs: Dict[str, Any],
        metric_kind: str
    ) -> List[Tuple[Dict[str, Any], str]]:
        """Generate boundary mutations (±1, ±2)."""
        mutations = []
        
        if metric_kind == "density":
            base = inputs.get("min_verified", 0)
            for delta in [-2, -1, 1, 2]:
                mutated = copy.deepcopy(inputs)
                mutated["min_verified"] = base + delta
                mutations.append((mutated, f"min_verified {'+' if delta > 0 else ''}{delta}"))
        
        elif metric_kind == "goal_hit":
            base = inputs.get("min_hits", 0)
            for delta in [-2, -1, 1, 2]:
                mutated = copy.deepcopy(inputs)
                mutated["min_hits"] = base + delta
                mutations.append((mutated, f"min_hits {'+' if delta > 0 else ''}{delta}"))
        
        elif metric_kind == "chain_length":
            base = inputs.get("min_length", 1)
            for delta in [-2, -1, 1, 2]:
                mutated = copy.deepcopy(inputs)
                mutated["min_length"] = max(0, base + delta)
                mutations.append((mutated, f"min_length {'+' if delta > 0 else ''}{delta}"))
        
        elif metric_kind == "multi_goal":
            required = inputs.get("required", set())
            if required:
                elem = self._rng.choice(list(required))
                mutated = copy.deepcopy(inputs)
                mutated["required"] = required - {elem}
                mutations.append((mutated, f"required -{elem}"))
            mutated = copy.deepcopy(inputs)
            mutated["required"] = required | {f"new_{self._rng.randint(0, 999)}"}
            mutations.append((mutated, "required +1 element"))
        
        return mutations


# ===========================================================================
# SHADOW METRICS
# ===========================================================================

class ShadowMetrics:
    """Shadow implementations for differential testing."""
    
    @staticmethod
    def compute_goal_hit_shadow(
        statements: List[Dict[str, Any]],
        targets: Set[str],
        min_hits: int,
    ) -> Tuple[bool, float]:
        hits = 0
        seen: Set[str] = set()
        for stmt in statements:
            h = stmt.get('hash')
            if h is not None and h not in seen:
                seen.add(h)
                if h in targets:
                    hits += 1
        return (hits >= min_hits), float(hits)
    
    @staticmethod
    def compute_sparse_success_shadow(
        verified_count: int,
        attempted_count: int,
        min_verified: int,
    ) -> Tuple[bool, float]:
        return (verified_count >= min_verified), float(verified_count)
    
    @staticmethod
    def compute_multi_goal_success_shadow(
        verified: Set[str],
        required: Set[str],
    ) -> Tuple[bool, float]:
        if not required:
            return True, 0.0
        met = sum(1 for g in required if g in verified)
        return (met == len(required)), float(met)


# ===========================================================================
# EQUIVALENCE ORACLE
# ===========================================================================

class EquivalenceOracle:
    """Compares production and shadow results."""
    
    @staticmethod
    def are_equivalent(
        prod: Tuple[bool, float],
        shadow: Tuple[bool, float],
        epsilon: float = 1e-9
    ) -> bool:
        if prod[0] != shadow[0]:
            return False
        
        if math.isnan(prod[1]) and math.isnan(shadow[1]):
            return True
        if math.isnan(prod[1]) or math.isnan(shadow[1]):
            return False
        
        if math.isinf(prod[1]) and math.isinf(shadow[1]):
            return (prod[1] > 0) == (shadow[1] > 0)
        
        return abs(prod[1] - shadow[1]) < epsilon


# ===========================================================================
# METRIC EXECUTOR
# ===========================================================================

class MetricExecutor:
    """Executes metric functions."""
    
    @staticmethod
    def execute_production(metric_kind: str, inputs: Dict[str, Any]) -> Tuple[bool, float]:
        if metric_kind == "goal_hit":
            return compute_goal_hit(
                inputs["statements"],
                inputs["targets"],
                inputs["min_hits"]
            )
        elif metric_kind == "density":
            return compute_sparse_success(
                inputs["verified_count"],
                inputs["attempted_count"],
                inputs["min_verified"]
            )
        elif metric_kind == "chain_length":
            return compute_chain_success(
                inputs["statements"],
                inputs["graph"],
                inputs["target"],
                inputs["min_length"]
            )
        elif metric_kind == "multi_goal":
            return compute_multi_goal_success(
                inputs["verified"],
                inputs["required"]
            )
        raise ValueError(f"Unknown metric kind: {metric_kind}")
    
    @staticmethod
    def execute_shadow(metric_kind: str, inputs: Dict[str, Any]) -> Optional[Tuple[bool, float]]:
        try:
            if metric_kind == "goal_hit":
                return ShadowMetrics.compute_goal_hit_shadow(
                    inputs["statements"],
                    inputs["targets"],
                    inputs["min_hits"]
                )
            elif metric_kind == "density":
                return ShadowMetrics.compute_sparse_success_shadow(
                    inputs["verified_count"],
                    inputs["attempted_count"],
                    inputs["min_verified"]
                )
            elif metric_kind == "multi_goal":
                return ShadowMetrics.compute_multi_goal_success_shadow(
                    inputs["verified"],
                    inputs["required"]
                )
        except Exception:
            return None
        return None


# ===========================================================================
# HARNESS RUNNER
# ===========================================================================

class AdversarialHarness:
    """Main harness for running adversarial tests."""
    
    def __init__(self, seed: int = SEED_HARNESS) -> None:
        self.input_gen = InputGenerator(seed)
        self.fault_injector = FaultInjector(seed)
        self.mutation_op = MutationOperator(seed)
        self.oracle = EquivalenceOracle()
        self.executor = MetricExecutor()
        self._seed = seed
    
    def run_fault_mode(
        self,
        metric_kind: str,
        n_tests: int
    ) -> HarnessSummary:
        """Run fault injection tests."""
        self.input_gen.reset()
        start = time.time()
        
        summary = HarnessSummary(
            mode="fault",
            metric_kind=metric_kind,
            total_cases=0,
            passed=0,
            failed=0,
            mismatches=0,
            errors=0,
        )
        
        fault_types = list(FaultType)
        tests_per_fault = max(1, n_tests // len(fault_types))
        
        for fault_type in fault_types:
            for _ in range(tests_per_fault):
                summary.total_cases += 1
                
                inputs = self.input_gen.generate_inputs(metric_kind)
                faulted, desc = self.fault_injector.inject_fault(
                    inputs, metric_kind, fault_type
                )
                
                try:
                    result = self.executor.execute_production(metric_kind, faulted)
                    summary.passed += 1
                except Exception as e:
                    summary.errors += 1
                    ft_name = fault_type.value
                    summary.fault_divergences[ft_name] = summary.fault_divergences.get(ft_name, 0) + 1
        
        summary.duration_seconds = time.time() - start
        return summary
    
    def run_mutation_mode(
        self,
        metric_kind: str,
        n_tests: int
    ) -> HarnessSummary:
        """Run mutation tests."""
        self.input_gen.reset()
        start = time.time()
        
        summary = HarnessSummary(
            mode="mutation",
            metric_kind=metric_kind,
            total_cases=0,
            passed=0,
            failed=0,
            mismatches=0,
            errors=0,
        )
        
        for _ in range(n_tests):
            inputs = self.input_gen.generate_inputs(metric_kind)
            
            try:
                base_result = self.executor.execute_production(metric_kind, inputs)
            except Exception:
                continue
            
            mutations = self.mutation_op.mutate_boundary(inputs, metric_kind)
            
            for mutated, desc in mutations:
                summary.total_cases += 1
                
                try:
                    mut_result = self.executor.execute_production(metric_kind, mutated)
                    
                    if mut_result != base_result:
                        summary.fault_divergences[desc] = summary.fault_divergences.get(desc, 0) + 1
                    
                    summary.passed += 1
                except Exception as e:
                    summary.errors += 1
        
        summary.duration_seconds = time.time() - start
        return summary
    
    def run_replay_mode(
        self,
        metric_kind: str,
        n_tests: int
    ) -> HarnessSummary:
        """Run replay determinism tests."""
        start = time.time()
        
        summary = HarnessSummary(
            mode="replay",
            metric_kind=metric_kind,
            total_cases=0,
            passed=0,
            failed=0,
            mismatches=0,
            errors=0,
        )
        
        self.input_gen.reset()
        run1_inputs = []
        run1_results = []
        
        for _ in range(n_tests):
            inputs = self.input_gen.generate_inputs(metric_kind)
            run1_inputs.append(copy.deepcopy(inputs))
            
            try:
                result = self.executor.execute_production(metric_kind, inputs)
                run1_results.append(result)
            except Exception as e:
                run1_results.append(None)
        
        self.input_gen.reset()
        
        for i in range(n_tests):
            summary.total_cases += 1
            inputs = self.input_gen.generate_inputs(metric_kind)
            
            if inputs != run1_inputs[i]:
                summary.mismatches += 1
                summary.fault_divergences["input_mismatch"] = summary.fault_divergences.get("input_mismatch", 0) + 1
                continue
            
            try:
                result = self.executor.execute_production(metric_kind, inputs)
                
                if run1_results[i] is None:
                    summary.errors += 1
                elif result != run1_results[i]:
                    summary.mismatches += 1
                    summary.fault_divergences["result_mismatch"] = summary.fault_divergences.get("result_mismatch", 0) + 1
                else:
                    summary.passed += 1
            except Exception:
                if run1_results[i] is None:
                    summary.passed += 1
                else:
                    summary.errors += 1
        
        summary.duration_seconds = time.time() - start
        return summary
    
    def run(
        self,
        metric_kind: str,
        n_tests: int,
        mode: str
    ) -> List[HarnessSummary]:
        """Run the harness in the specified mode."""
        summaries = []
        
        if mode == "all":
            modes_to_run = ["fault", "mutation", "replay"]
        else:
            modes_to_run = [mode]
        
        for m in modes_to_run:
            if m == "fault":
                summaries.append(self.run_fault_mode(metric_kind, n_tests))
            elif m == "mutation":
                summaries.append(self.run_mutation_mode(metric_kind, n_tests))
            elif m == "replay":
                summaries.append(self.run_replay_mode(metric_kind, n_tests))
        
        return summaries
    
    def run_with_profile(
        self,
        metric_kind: str,
        profile: str
    ) -> List[HarnessSummary]:
        """Run harness using a predefined profile."""
        config = PROFILE_CONFIG[profile]
        n_tests = config["n_tests"]
        modes = config["modes"]
        
        summaries = []
        for mode in modes:
            summaries.extend(self.run(metric_kind, n_tests, mode))
        
        return summaries


# ===========================================================================
# COVERAGE REPORTER
# ===========================================================================

def generate_coverage_report() -> CoverageReport:
    """
    Generate a coverage report for CI decoration.
    
    Contract: Output includes all COVERAGE_REPORT_REQUIRED_KEYS.
    """
    fault_injector = FaultInjector(SEED_HARNESS)
    
    fault_types_per_metric: Dict[str, List[str]] = {}
    mutation_categories_per_metric: Dict[str, List[str]] = {}
    replay_sizes_per_metric: Dict[str, int] = {}
    
    for metric_kind in METRIC_KINDS:
        applicable_faults = fault_injector.get_applicable_faults(metric_kind)
        fault_types_per_metric[metric_kind] = [ft.value for ft in applicable_faults]
        
        mutation_categories_per_metric[metric_kind] = MUTATION_CATEGORIES.get(metric_kind, [])
        
        replay_sizes_per_metric[metric_kind] = PROFILE_CONFIG["standard"]["n_tests"]
    
    all_faults = set()
    for faults in fault_types_per_metric.values():
        all_faults.update(faults)
    
    return CoverageReport(
        metric_kinds=METRIC_KINDS,
        fault_types_per_metric=fault_types_per_metric,
        mutation_categories_per_metric=mutation_categories_per_metric,
        replay_sizes_per_metric=replay_sizes_per_metric,
        total_fault_types=len(all_faults),
        total_metrics=len(METRIC_KINDS),
    )


def coverage_report_to_json(report: CoverageReport) -> str:
    """
    Convert coverage report to JSON string.
    
    Contract: Output contains all COVERAGE_REPORT_REQUIRED_KEYS.
    """
    data = {
        "metric_kinds": report.metric_kinds,
        "fault_types_per_metric": report.fault_types_per_metric,
        "mutation_categories_per_metric": report.mutation_categories_per_metric,
        "replay_sizes_per_metric": report.replay_sizes_per_metric,
        "total_fault_types": report.total_fault_types,
        "total_metrics": report.total_metrics,
    }
    
    # Validate contract
    for key in COVERAGE_REPORT_REQUIRED_KEYS:
        assert key in data, f"Coverage report missing required key: {key}"
    
    return json.dumps(data, indent=2, sort_keys=True)


def validate_coverage_report(json_str: str) -> Tuple[bool, List[str]]:
    """
    Validate a coverage report JSON meets the contract.
    
    Returns: (valid, list of violations)
    """
    violations = []
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    
    # Check required keys
    for key in COVERAGE_REPORT_REQUIRED_KEYS:
        if key not in data:
            violations.append(f"Missing required key: {key}")
    
    # Check metric_kinds not empty
    if not data.get("metric_kinds"):
        violations.append("metric_kinds is empty")
    
    # Check all metrics have fault types
    faults = data.get("fault_types_per_metric", {})
    for metric in data.get("metric_kinds", []):
        if metric not in faults or not faults[metric]:
            violations.append(f"No fault types for metric: {metric}")
    
    # Check all metrics have mutation categories
    mutations = data.get("mutation_categories_per_metric", {})
    for metric in data.get("metric_kinds", []):
        if metric not in mutations or not mutations[metric]:
            violations.append(f"No mutation categories for metric: {metric}")
    
    return len(violations) == 0, violations


# ===========================================================================
# REGRESSION RADAR — Fixed Battery
# ===========================================================================

def _hash_result(result: Tuple[bool, float]) -> str:
    """Hash a metric result for comparison."""
    data = f"{result[0]}:{result[1]}"
    return hashlib.sha256(data.encode()).hexdigest()[:12]


def run_regression_radar(seed: int = SEED_HARNESS) -> List[RegressionResult]:
    """
    Run regression radar: fixed battery for all metrics.
    
    Contract:
    - Returns list of RegressionResult
    - Each result has: metric_kind, test_name, expected_hash, actual_hash, match
    - Deterministic: same seed → same results
    """
    results: List[RegressionResult] = []
    executor = MetricExecutor()
    
    # Goal hit: boundary tests
    statements = [{"hash": f"h{i}"} for i in range(5)]
    targets = {f"h{i}" for i in range(5)}
    
    goal_hit_result = executor.execute_production("goal_hit", {
        "statements": statements,
        "targets": targets,
        "min_hits": 5,
    })
    expected = (True, 5.0)
    results.append(RegressionResult(
        metric_kind="goal_hit",
        test_name="boundary_at_threshold",
        expected_hash=_hash_result(expected),
        actual_hash=_hash_result(goal_hit_result),
        match=goal_hit_result == expected,
    ))
    
    goal_hit_result2 = executor.execute_production("goal_hit", {
        "statements": statements,
        "targets": targets,
        "min_hits": 6,
    })
    expected2 = (False, 5.0)
    results.append(RegressionResult(
        metric_kind="goal_hit",
        test_name="boundary_below_threshold",
        expected_hash=_hash_result(expected2),
        actual_hash=_hash_result(goal_hit_result2),
        match=goal_hit_result2 == expected2,
    ))
    
    # Density: boundary tests
    density_result = executor.execute_production("density", {
        "verified_count": 50,
        "attempted_count": 100,
        "min_verified": 50,
    })
    expected_density = (True, 50.0)
    results.append(RegressionResult(
        metric_kind="density",
        test_name="boundary_at_threshold",
        expected_hash=_hash_result(expected_density),
        actual_hash=_hash_result(density_result),
        match=density_result == expected_density,
    ))
    
    density_result2 = executor.execute_production("density", {
        "verified_count": 49,
        "attempted_count": 100,
        "min_verified": 50,
    })
    expected_density2 = (False, 49.0)
    results.append(RegressionResult(
        metric_kind="density",
        test_name="boundary_below_threshold",
        expected_hash=_hash_result(expected_density2),
        actual_hash=_hash_result(density_result2),
        match=density_result2 == expected_density2,
    ))
    
    # Chain length: linear chain
    chain_statements = [{"hash": f"h{i}"} for i in range(5)]
    chain_graph = {f"h{i}": [f"h{i-1}"] for i in range(1, 5)}
    chain_result = executor.execute_production("chain_length", {
        "statements": chain_statements,
        "graph": chain_graph,
        "target": "h4",
        "min_length": 5,
    })
    expected_chain = (True, 5.0)
    results.append(RegressionResult(
        metric_kind="chain_length",
        test_name="linear_chain_5",
        expected_hash=_hash_result(expected_chain),
        actual_hash=_hash_result(chain_result),
        match=chain_result == expected_chain,
    ))
    
    broken_statements = [{"hash": "h0"}, {"hash": "h2"}, {"hash": "h4"}]
    chain_result2 = executor.execute_production("chain_length", {
        "statements": broken_statements,
        "graph": chain_graph,
        "target": "h4",
        "min_length": 5,
    })
    expected_chain2 = (False, 1.0)
    results.append(RegressionResult(
        metric_kind="chain_length",
        test_name="broken_chain",
        expected_hash=_hash_result(expected_chain2),
        actual_hash=_hash_result(chain_result2),
        match=chain_result2 == expected_chain2,
    ))
    
    # Multi goal: all met
    multi_result = executor.execute_production("multi_goal", {
        "verified": {"h1", "h2", "h3", "h4", "h5"},
        "required": {"h1", "h2", "h3"},
    })
    expected_multi = (True, 3.0)
    results.append(RegressionResult(
        metric_kind="multi_goal",
        test_name="all_goals_met",
        expected_hash=_hash_result(expected_multi),
        actual_hash=_hash_result(multi_result),
        match=multi_result == expected_multi,
    ))
    
    multi_result2 = executor.execute_production("multi_goal", {
        "verified": {"h1", "h2"},
        "required": {"h1", "h2", "h3"},
    })
    expected_multi2 = (False, 2.0)
    results.append(RegressionResult(
        metric_kind="multi_goal",
        test_name="partial_goals_met",
        expected_hash=_hash_result(expected_multi2),
        actual_hash=_hash_result(multi_result2),
        match=multi_result2 == expected_multi2,
    ))
    
    # High-volume determinism check (100 runs per metric)
    input_gen = InputGenerator(seed)
    for metric_kind in METRIC_KINDS:
        input_gen.reset()
        results_run1 = []
        for _ in range(100):
            inputs = input_gen.generate_inputs(metric_kind)
            try:
                result = executor.execute_production(metric_kind, inputs)
                results_run1.append(_hash_result(result))
            except Exception:
                results_run1.append("ERROR")
        
        input_gen.reset()
        results_run2 = []
        for _ in range(100):
            inputs = input_gen.generate_inputs(metric_kind)
            try:
                result = executor.execute_production(metric_kind, inputs)
                results_run2.append(_hash_result(result))
            except Exception:
                results_run2.append("ERROR")
        
        seq_hash1 = hashlib.sha256("".join(results_run1).encode()).hexdigest()[:12]
        seq_hash2 = hashlib.sha256("".join(results_run2).encode()).hexdigest()[:12]
        
        results.append(RegressionResult(
            metric_kind=metric_kind,
            test_name="high_volume_determinism_100",
            expected_hash=seq_hash1,
            actual_hash=seq_hash2,
            match=seq_hash1 == seq_hash2,
        ))
    
    return results


def print_regression_radar_results(results: List[RegressionResult], quiet: bool = False) -> int:
    """
    Print regression radar results.
    
    Contract:
    - On success: Single line "REGRESSION RADAR: OK", exit code 0
    - On mismatch: Full diff with metric name, test label, both hashes, exit code 1
    """
    all_match = all(r.match for r in results)
    
    if all_match:
        # Quiet on success (embeddable in pre-commit)
        print("REGRESSION RADAR: OK")
        if not quiet:
            print(f"  {len(results)} tests passed")
        return 0
    else:
        # Loud on mismatch
        print("REGRESSION RADAR: MISMATCH")
        print("")
        for r in results:
            if not r.match:
                print(f"  [{r.metric_kind}] {r.test_name}")
                print(f"    Expected hash: {r.expected_hash}")
                print(f"    Actual hash:   {r.actual_hash}")
                print("")
        
        # Summary
        mismatches = [r for r in results if not r.match]
        print(f"  {len(mismatches)} of {len(results)} tests failed")
        return 1


# ===========================================================================
# METRIC ROBUSTNESS SCORECARD — Advisory Report
# ===========================================================================

SCORECARD_SCHEMA_VERSION = "1.0.0"


@dataclass
class MetricScorecard:
    """Robustness scorecard for a single metric."""
    metric_kind: str
    scenarios_covered: List[str]
    fault_types_seen: int
    mutation_categories_seen: int
    replay_regressions_detected: int
    total_tests: int
    total_passed: int
    total_errors: int


def build_robustness_scorecard(
    summaries: List[HarnessSummary],
    scenarios_run: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Build a read-only scorecard summarizing adversarial health.
    
    This is purely observational — no CI exit codes, just data.
    
    Args:
        summaries: List of HarnessSummary from harness runs
        scenarios_run: Optional list of scenario names that were executed
    
    Returns:
        Dict with scorecard data (deterministic, stable JSON)
    """
    # Aggregate by metric kind
    metrics_data: Dict[str, Dict[str, Any]] = {}
    
    for metric_kind in METRIC_KINDS:
        metrics_data[metric_kind] = {
            "scenarios_covered": [],
            "fault_types_seen": 0,
            "mutation_categories_seen": 0,
            "replay_regressions_detected": 0,
            "total_tests": 0,
            "total_passed": 0,
            "total_errors": 0,
            "modes_run": set(),
        }
    
    # Process summaries
    for summary in summaries:
        mk = summary.metric_kind
        if mk not in metrics_data:
            continue
        
        data = metrics_data[mk]
        data["total_tests"] += summary.total_cases
        data["total_passed"] += summary.passed
        data["total_errors"] += summary.errors
        data["modes_run"].add(summary.mode)
        
        # Count unique fault types from divergences
        if summary.mode == "fault":
            data["fault_types_seen"] = len(summary.fault_divergences)
        elif summary.mode == "mutation":
            data["mutation_categories_seen"] = len(summary.fault_divergences)
        elif summary.mode == "replay":
            data["replay_regressions_detected"] += summary.mismatches
    
    # Determine scenarios covered per metric
    if scenarios_run:
        for scenario_name in scenarios_run:
            if scenario_name in SCENARIOS:
                scenario = SCENARIOS[scenario_name]
                for mk in scenario.metric_kinds:
                    if mk in metrics_data:
                        if scenario_name not in metrics_data[mk]["scenarios_covered"]:
                            metrics_data[mk]["scenarios_covered"].append(scenario_name)
    
    # Build final scorecard
    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "generated_seed": SEED_HARNESS,
        "metrics": {},
    }
    
    for mk in sorted(METRIC_KINDS):
        data = metrics_data[mk]
        scorecard["metrics"][mk] = {
            "scenarios_covered": sorted(data["scenarios_covered"]),
            "fault_types_seen": data["fault_types_seen"],
            "mutation_categories_seen": data["mutation_categories_seen"],
            "replay_regressions_detected": data["replay_regressions_detected"],
            "total_tests": data["total_tests"],
            "total_passed": data["total_passed"],
            "total_errors": data["total_errors"],
            "modes_run": sorted(data["modes_run"]),
        }
    
    return scorecard


def scorecard_to_json(scorecard: Dict[str, Any]) -> str:
    """Convert scorecard to deterministic JSON string."""
    return json.dumps(scorecard, indent=2, sort_keys=True)


# ===========================================================================
# PHASE III — RISK ANALYTICS ENGINE
# ===========================================================================

# Risk level thresholds (rule-based)
RISK_THRESHOLDS = {
    "regression_high": 3,      # >= 3 replay regressions → HIGH
    "regression_medium": 1,    # >= 1 replay regression → MEDIUM
    "fault_coverage_low": 3,   # < 3 fault types → sparse coverage
    "mutation_coverage_low": 2, # < 2 mutation categories → sparse
}

# Robustness tags
ROBUSTNESS_TAGS = {
    "well_exercised": "WELL_EXERCISED",
    "partially_tested": "PARTIALLY_TESTED", 
    "sparsely_tested": "SPARSELY_TESTED",
}

# Health statuses
HEALTH_STATUS = {
    "ok": "OK",
    "warn": "WARN",
    "attention": "ATTENTION",
}


def summarize_scenario_risk(
    scorecard: Dict[str, Any],
    scenario_name: str
) -> Dict[str, Any]:
    """
    Summarize risk for a specific scenario.
    
    Args:
        scorecard: A scorecard dict from build_robustness_scorecard()
        scenario_name: Name of the scenario to analyze
    
    Returns:
        Dict with:
        - schema_version
        - scenario_name
        - metrics_covered
        - fault_coverage (count of fault types exercised)
        - mutation_coverage (count of mutation categories)
        - replay_regressions_detected
        - risk_level: "LOW" | "MEDIUM" | "HIGH"
    """
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    scenario = SCENARIOS[scenario_name]
    metrics = scorecard.get("metrics", {})
    
    # Aggregate across metrics covered by this scenario
    metrics_covered = []
    total_fault_types = 0
    total_mutation_categories = 0
    total_regressions = 0
    
    for mk in scenario.metric_kinds:
        if mk in metrics:
            metrics_covered.append(mk)
            metric_data = metrics[mk]
            
            # Check if this scenario is in the metric's covered scenarios
            if scenario_name in metric_data.get("scenarios_covered", []):
                total_fault_types += metric_data.get("fault_types_seen", 0)
                total_mutation_categories += metric_data.get("mutation_categories_seen", 0)
                total_regressions += metric_data.get("replay_regressions_detected", 0)
    
    # Calculate risk level (rule-based)
    if total_regressions >= RISK_THRESHOLDS["regression_high"]:
        risk_level = "HIGH"
    elif total_regressions >= RISK_THRESHOLDS["regression_medium"]:
        risk_level = "MEDIUM"
    elif total_fault_types < RISK_THRESHOLDS["fault_coverage_low"]:
        risk_level = "MEDIUM"  # Sparse coverage is a risk signal
    else:
        risk_level = "LOW"
    
    return {
        "schema_version": scorecard.get("schema_version", SCORECARD_SCHEMA_VERSION),
        "scenario_name": scenario_name,
        "scenario_profile": scenario.profile,
        "metrics_covered": sorted(metrics_covered),
        "fault_coverage": total_fault_types,
        "mutation_coverage": total_mutation_categories,
        "replay_regressions_detected": total_regressions,
        "risk_level": risk_level,
    }


def build_metric_robustness_radar(
    scorecards: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build a multi-scenario robustness radar for all metrics.
    
    Aggregates multiple scorecards to provide a holistic view of
    metric robustness across different scenarios.
    
    Args:
        scorecards: Sequence of scorecard dicts from build_robustness_scorecard()
    
    Returns:
        Dict with:
        - schema_version
        - metrics: Dict mapping metric_kind to:
            - scenarios_exercised: list of scenario names
            - scenarios_with_regressions: list of scenario names with regressions
            - total_tests: total tests across all scorecards
            - total_regressions: total regressions detected
            - robustness_tag: "WELL_EXERCISED" | "PARTIALLY_TESTED" | "SPARSELY_TESTED"
        - global:
            - metrics_at_risk: list of metrics with regressions in multiple scenarios
            - total_scenarios_analyzed: count
    """
    # Aggregate by metric
    metrics_agg: Dict[str, Dict[str, Any]] = {}
    
    for mk in METRIC_KINDS:
        metrics_agg[mk] = {
            "scenarios_exercised": set(),
            "scenarios_with_regressions": set(),
            "total_tests": 0,
            "total_regressions": 0,
        }
    
    # Process each scorecard
    for scorecard in scorecards:
        metrics = scorecard.get("metrics", {})
        
        for mk in METRIC_KINDS:
            if mk not in metrics:
                continue
            
            metric_data = metrics[mk]
            agg = metrics_agg[mk]
            
            # Track scenarios exercised
            for sc_name in metric_data.get("scenarios_covered", []):
                agg["scenarios_exercised"].add(sc_name)
            
            # Track tests and regressions
            agg["total_tests"] += metric_data.get("total_tests", 0)
            regressions = metric_data.get("replay_regressions_detected", 0)
            agg["total_regressions"] += regressions
            
            # Track which scenarios had regressions
            if regressions > 0:
                for sc_name in metric_data.get("scenarios_covered", []):
                    agg["scenarios_with_regressions"].add(sc_name)
    
    # Build final radar
    radar = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "metrics": {},
        "global": {
            "metrics_at_risk": [],
            "total_scenarios_analyzed": 0,
        },
    }
    
    all_scenarios = set()
    
    for mk in sorted(METRIC_KINDS):
        agg = metrics_agg[mk]
        scenarios_exercised = sorted(agg["scenarios_exercised"])
        scenarios_with_regressions = sorted(agg["scenarios_with_regressions"])
        
        all_scenarios.update(agg["scenarios_exercised"])
        
        # Determine robustness tag
        num_scenarios = len(scenarios_exercised)
        if num_scenarios >= 3:
            robustness_tag = ROBUSTNESS_TAGS["well_exercised"]
        elif num_scenarios >= 1:
            robustness_tag = ROBUSTNESS_TAGS["partially_tested"]
        else:
            robustness_tag = ROBUSTNESS_TAGS["sparsely_tested"]
        
        radar["metrics"][mk] = {
            "scenarios_exercised": scenarios_exercised,
            "scenarios_with_regressions": scenarios_with_regressions,
            "total_tests": agg["total_tests"],
            "total_regressions": agg["total_regressions"],
            "robustness_tag": robustness_tag,
        }
        
        # Track metrics at risk (regressions in multiple scenarios)
        if len(scenarios_with_regressions) >= 2:
            radar["global"]["metrics_at_risk"].append(mk)
    
    radar["global"]["total_scenarios_analyzed"] = len(all_scenarios)
    radar["global"]["metrics_at_risk"] = sorted(radar["global"]["metrics_at_risk"])
    
    return radar


def summarize_adversarial_health_for_global_health(
    radar: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Summarize adversarial health for dashboard and MAAS consumption.
    
    Provides a small, consumable block for global health monitoring.
    
    Args:
        radar: A radar dict from build_metric_robustness_radar()
    
    Returns:
        Dict with:
        - adversarial_coverage_ok: bool
        - metrics_at_risk: list of metric names
        - status: "OK" | "WARN" | "ATTENTION"
        - summary: human-readable one-liner
    """
    metrics = radar.get("metrics", {})
    global_data = radar.get("global", {})
    
    metrics_at_risk = global_data.get("metrics_at_risk", [])
    total_scenarios = global_data.get("total_scenarios_analyzed", 0)
    
    # Count metrics by robustness tag
    well_exercised = 0
    partially_tested = 0
    sparsely_tested = 0
    total_regressions = 0
    
    for mk, data in metrics.items():
        tag = data.get("robustness_tag", "")
        if tag == ROBUSTNESS_TAGS["well_exercised"]:
            well_exercised += 1
        elif tag == ROBUSTNESS_TAGS["partially_tested"]:
            partially_tested += 1
        else:
            sparsely_tested += 1
        total_regressions += data.get("total_regressions", 0)
    
    # Determine status
    if len(metrics_at_risk) > 0:
        # Multiple scenarios show regressions for some metrics
        status = HEALTH_STATUS["attention"]
        adversarial_coverage_ok = False
        summary = f"{len(metrics_at_risk)} metric(s) at risk with cross-scenario regressions"
    elif total_regressions > 0:
        # Some regressions but not across multiple scenarios
        status = HEALTH_STATUS["warn"]
        adversarial_coverage_ok = True
        summary = f"{total_regressions} regression(s) detected, isolated to single scenarios"
    elif sparsely_tested > 0:
        # Some metrics lack coverage
        status = HEALTH_STATUS["warn"]
        adversarial_coverage_ok = True
        summary = f"{sparsely_tested} metric(s) sparsely tested"
    else:
        status = HEALTH_STATUS["ok"]
        adversarial_coverage_ok = True
        summary = f"All {len(METRIC_KINDS)} metrics exercised across {total_scenarios} scenarios"
    
    return {
        "adversarial_coverage_ok": adversarial_coverage_ok,
        "metrics_at_risk": metrics_at_risk,
        "status": status,
        "summary": summary,
        "details": {
            "total_metrics": len(METRIC_KINDS),
            "well_exercised": well_exercised,
            "partially_tested": partially_tested,
            "sparsely_tested": sparsely_tested,
            "total_regressions": total_regressions,
            "total_scenarios_analyzed": total_scenarios,
        },
    }


def health_summary_to_json(health: Dict[str, Any]) -> str:
    """Convert health summary to deterministic JSON string."""
    return json.dumps(health, indent=2, sort_keys=True)


# ===========================================================================
# PHASE IV — ADVERSARIAL COVERAGE PROMOTION GATE & EARLY WARNING RADAR
# ===========================================================================

# Coverage status values
COVERAGE_STATUS = {
    "ok": "OK",
    "sparse": "SPARSE",
    "at_risk": "AT_RISK",
}

# Promotion readiness statuses
PROMOTION_STATUS = {
    "ok": "OK",
    "warn": "WARN",
    "block": "BLOCK",
}

# Director panel status lights
STATUS_LIGHT = {
    "green": "GREEN",
    "yellow": "YELLOW",
    "red": "RED",
}

# Core uplift metrics (used for promotion blocking)
CORE_UPLIFT_METRICS = ["goal_hit", "density"]


def build_metric_adversarial_coverage_index(
    radar: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build a structured adversarial coverage index for each metric.
    
    Converts robustness radar into a coverage index with explicit
    coverage_status per metric for promotion gating.
    
    Args:
        radar: A radar dict from build_metric_robustness_radar()
    
    Returns:
        Dict with:
        - schema_version
        - metrics: Dict mapping metric_kind to:
            - robustness_tag: WELL_EXERCISED | PARTIALLY_TESTED | SPARSELY_TESTED
            - scenario_count: number of scenarios exercised
            - regression_count: total regressions detected
            - coverage_status: OK | SPARSE | AT_RISK
        - global:
            - metrics_at_risk: list of metrics with AT_RISK status
            - coverage_ok: bool (True if all metrics OK or SPARSE, False if any AT_RISK)
    """
    metrics = radar.get("metrics", {})
    global_data = radar.get("global", {})
    metrics_at_risk_global = set(global_data.get("metrics_at_risk", []))
    
    coverage_index = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "metrics": {},
        "global": {
            "metrics_at_risk": [],
            "coverage_ok": True,
        },
    }
    
    for mk in sorted(METRIC_KINDS):
        if mk not in metrics:
            # Metric not in radar → SPARSELY_TESTED, AT_RISK
            coverage_index["metrics"][mk] = {
                "robustness_tag": ROBUSTNESS_TAGS["sparsely_tested"],
                "scenario_count": 0,
                "regression_count": 0,
                "coverage_status": COVERAGE_STATUS["at_risk"],
            }
            coverage_index["global"]["metrics_at_risk"].append(mk)
            coverage_index["global"]["coverage_ok"] = False
            continue
        
        metric_data = metrics[mk]
        robustness_tag = metric_data.get("robustness_tag", ROBUSTNESS_TAGS["sparsely_tested"])
        scenario_count = len(metric_data.get("scenarios_exercised", []))
        regression_count = metric_data.get("total_regressions", 0)
        
        # Determine coverage_status
        if mk in metrics_at_risk_global:
            # Regressions in multiple scenarios → AT_RISK
            coverage_status = COVERAGE_STATUS["at_risk"]
            coverage_index["global"]["metrics_at_risk"].append(mk)
            coverage_index["global"]["coverage_ok"] = False
        elif scenario_count == 0:
            # No scenarios exercised → AT_RISK (truly missing)
            coverage_status = COVERAGE_STATUS["at_risk"]
            coverage_index["global"]["metrics_at_risk"].append(mk)
            coverage_index["global"]["coverage_ok"] = False
        elif robustness_tag == ROBUSTNESS_TAGS["sparsely_tested"]:
            # Sparse coverage → SPARSE
            coverage_status = COVERAGE_STATUS["sparse"]
        elif scenario_count < 2:
            # Only 1 scenario → SPARSE (insufficient coverage)
            coverage_status = COVERAGE_STATUS["sparse"]
        elif regression_count > 0:
            # Has regressions but isolated → SPARSE (not blocking but concerning)
            coverage_status = COVERAGE_STATUS["sparse"]
        else:
            # Well exercised, no regressions → OK
            coverage_status = COVERAGE_STATUS["ok"]
        
        coverage_index["metrics"][mk] = {
            "robustness_tag": robustness_tag,
            "scenario_count": scenario_count,
            "regression_count": regression_count,
            "coverage_status": coverage_status,
        }
    
    coverage_index["global"]["metrics_at_risk"] = sorted(coverage_index["global"]["metrics_at_risk"])
    
    return coverage_index


def evaluate_adversarial_readiness_for_promotion(
    coverage_index: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate promotion readiness from adversarial coverage perspective.
    
    Determines if metrics are ready for promotion based on adversarial
    coverage status. Provides blocking signals for core uplift metrics.
    
    Args:
        coverage_index: A coverage index from build_metric_adversarial_coverage_index()
    
    Returns:
        Dict with:
        - promotion_ok: bool
        - metrics_blocking_promotion: list of metric names
        - status: OK | WARN | BLOCK
        - reasons: list of short neutral strings explaining status
    """
    metrics = coverage_index.get("metrics", {})
    global_data = coverage_index.get("global", {})
    
    metrics_blocking = []
    metrics_warning = []
    reasons = []
    
    # Check each metric
    for mk in METRIC_KINDS:
        if mk not in metrics:
            # Missing metric → blocking if core
            if mk in CORE_UPLIFT_METRICS:
                metrics_blocking.append(mk)
                reasons.append(f"{mk}: missing from coverage index")
            else:
                metrics_warning.append(mk)
                reasons.append(f"{mk}: missing from coverage index")
            continue
        
        metric_data = metrics[mk]
        coverage_status = metric_data.get("coverage_status", COVERAGE_STATUS["at_risk"])
        
        if coverage_status == COVERAGE_STATUS["at_risk"]:
            if mk in CORE_UPLIFT_METRICS:
                metrics_blocking.append(mk)
                reasons.append(f"{mk}: AT_RISK status (regressions in multiple scenarios)")
            else:
                metrics_warning.append(mk)
                reasons.append(f"{mk}: AT_RISK status")
        elif coverage_status == COVERAGE_STATUS["sparse"]:
            metrics_warning.append(mk)
            reasons.append(f"{mk}: SPARSE coverage")
    
    # Determine overall status
    if len(metrics_blocking) > 0:
        status = PROMOTION_STATUS["block"]
        promotion_ok = False
    elif len(metrics_warning) > 0:
        status = PROMOTION_STATUS["warn"]
        promotion_ok = True  # Warning but not blocking
    else:
        status = PROMOTION_STATUS["ok"]
        promotion_ok = True
    
    return {
        "promotion_ok": promotion_ok,
        "metrics_blocking_promotion": sorted(metrics_blocking),
        "metrics_warning": sorted(metrics_warning),
        "status": status,
        "reasons": sorted(reasons),
    }


def build_adversarial_director_panel(
    coverage_index: Dict[str, Any],
    readiness_eval: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build a high-level director panel for decision-making.
    
    Provides a concise, executive-friendly view of adversarial coverage
    status with clear status lights and neutral headlines.
    
    Args:
        coverage_index: A coverage index from build_metric_adversarial_coverage_index()
        readiness_eval: A readiness eval from evaluate_adversarial_readiness_for_promotion()
    
    Returns:
        Dict with:
        - status_light: GREEN | YELLOW | RED
        - adversarial_coverage_ok: bool
        - metrics_at_risk: list of metric names
        - headline: short neutral sentence (no "good/bad" language)
    """
    global_data = coverage_index.get("global", {})
    metrics_at_risk = global_data.get("metrics_at_risk", [])
    coverage_ok = global_data.get("coverage_ok", False)
    
    promotion_status = readiness_eval.get("status", PROMOTION_STATUS["block"])
    metrics_blocking = readiness_eval.get("metrics_blocking_promotion", [])
    
    # Determine status light
    if promotion_status == PROMOTION_STATUS["block"]:
        status_light = STATUS_LIGHT["red"]
    elif promotion_status == PROMOTION_STATUS["warn"]:
        status_light = STATUS_LIGHT["yellow"]
    else:
        status_light = STATUS_LIGHT["green"]
    
    # Build headline (neutral, factual)
    if len(metrics_blocking) > 0:
        headline = f"{len(metrics_blocking)} core metric(s) show cross-scenario regressions"
    elif len(metrics_at_risk) > 0:
        headline = f"{len(metrics_at_risk)} metric(s) flagged with cross-scenario regressions"
    elif not coverage_ok:
        headline = "Adversarial coverage incomplete for some metrics"
    else:
        num_metrics = len(METRIC_KINDS)
        num_scenarios = coverage_index.get("global", {}).get("total_scenarios_analyzed", 0)
        if num_scenarios > 0:
            headline = f"All {num_metrics} metrics exercised across {num_scenarios} adversarial scenarios"
        else:
            headline = f"All {num_metrics} metrics present in coverage index"
    
    return {
        "status_light": status_light,
        "adversarial_coverage_ok": coverage_ok,
        "metrics_at_risk": sorted(metrics_at_risk),
        "headline": headline,
        "promotion_status": promotion_status,
        "metrics_blocking_promotion": sorted(metrics_blocking),
    }


def coverage_index_to_json(coverage_index: Dict[str, Any]) -> str:
    """Convert coverage index to deterministic JSON string."""
    return json.dumps(coverage_index, indent=2, sort_keys=True)


def readiness_eval_to_json(readiness: Dict[str, Any]) -> str:
    """Convert readiness evaluation to deterministic JSON string."""
    return json.dumps(readiness, indent=2, sort_keys=True)


def director_panel_to_json(panel: Dict[str, Any]) -> str:
    """Convert director panel to deterministic JSON string."""
    return json.dumps(panel, indent=2, sort_keys=True)


# ===========================================================================
# ADVERSARIAL CURRICULUM DESIGNER & FAILOVER PLANNER
# ===========================================================================

# Complexity bands for scenario suggestions
COMPLEXITY_BANDS = {
    "low": "low",
    "medium": "medium",
    "high": "high",
}


def propose_adversarial_scenarios(
    coverage_index: Dict[str, Any],
    robustness_radar: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Propose new adversarial scenarios based on coverage gaps.
    
    Analyzes under-tested and at-risk metrics to suggest new scenario
    profiles that would improve coverage.
    
    Args:
        coverage_index: A coverage index from build_metric_adversarial_coverage_index()
        robustness_radar: A radar from build_metric_robustness_radar()
    
    Returns:
        Dict with:
        - metrics_needing_new_scenarios: list of metric names with low coverage
        - suggested_scenario_profiles: dict mapping scenario name -> profile info
        - neutral_notes: list of neutral observation strings
    """
    metrics = coverage_index.get("metrics", {})
    radar_metrics = robustness_radar.get("metrics", {})
    
    metrics_needing = []
    suggested_profiles: Dict[str, Dict[str, Any]] = {}
    notes = []
    
    # Identify metrics needing new scenarios
    for mk in METRIC_KINDS:
        metric_data = metrics.get(mk, {})
        coverage_status = metric_data.get("coverage_status", COVERAGE_STATUS["at_risk"])
        scenario_count = metric_data.get("scenario_count", 0)
        regression_count = metric_data.get("regression_count", 0)
        
        # Metrics that need more scenarios
        needs_more = False
        reason = ""
        
        if coverage_status == COVERAGE_STATUS["at_risk"]:
            needs_more = True
            reason = "AT_RISK status"
        elif coverage_status == COVERAGE_STATUS["sparse"]:
            needs_more = True
            reason = "SPARSE coverage"
        elif scenario_count < 2:
            needs_more = True
            reason = f"Only {scenario_count} scenario(s) exercised"
        elif regression_count > 0 and scenario_count < 3:
            needs_more = True
            reason = f"Regressions detected with limited scenario coverage"
        
        if needs_more:
            metrics_needing.append(mk)
            
            # Generate scenario name
            scenario_name = f"{mk}_adversarial_{len(suggested_profiles) + 1}"
            
            # Determine profile based on metric status
            if coverage_status == COVERAGE_STATUS["at_risk"]:
                profile = "full"  # Aggressive testing for at-risk metrics
                complexity = COMPLEXITY_BANDS["high"]
            elif regression_count > 0:
                profile = "standard"  # Standard for metrics with regressions
                complexity = COMPLEXITY_BANDS["medium"]
            else:
                profile = "standard"  # Standard for sparse coverage
                complexity = COMPLEXITY_BANDS["medium"]
            
            # Determine modes based on what's missing
            radar_data = radar_metrics.get(mk, {})
            scenarios_exercised = set(radar_data.get("scenarios_exercised", []))
            
            # Check existing scenarios to see what modes are used
            existing_modes = set()
            for sc_name in scenarios_exercised:
                if sc_name in SCENARIOS:
                    existing_modes.update(SCENARIOS[sc_name].modes)
            
            # Suggest modes that might be missing
            if "mutation" not in existing_modes:
                suggested_modes = ["fault", "mutation", "replay"]
            elif "replay" not in existing_modes:
                suggested_modes = ["fault", "replay"]
            else:
                suggested_modes = ["fault", "mutation", "replay"]  # All modes for comprehensive coverage
            
            suggested_profiles[scenario_name] = {
                "profile": profile,
                "metric_kinds": [mk],
                "modes": suggested_modes,
                "complexity_band": complexity,
                "rationale": f"Addresses {reason} for {mk}",
            }
            
            notes.append(f"{mk}: {reason} - suggests {profile} profile scenario")
    
    # Add multi-metric scenarios if multiple metrics need coverage
    if len(metrics_needing) >= 2:
        multi_name = f"multi_metric_gap_{len(suggested_profiles) + 1}"
        suggested_profiles[multi_name] = {
            "profile": "standard",
            "metric_kinds": sorted(metrics_needing[:3]),  # Up to 3 metrics
            "modes": ["fault", "mutation", "replay"],
            "complexity_band": COMPLEXITY_BANDS["medium"],
            "rationale": f"Addresses coverage gaps for {len(metrics_needing)} metrics",
        }
        notes.append(f"Multi-metric scenario suggested for {len(metrics_needing)} metrics")
    
    return {
        "metrics_needing_new_scenarios": sorted(metrics_needing),
        "suggested_scenario_profiles": suggested_profiles,
        "neutral_notes": sorted(notes),
    }


def build_adversarial_failover_plan(
    coverage_index: Dict[str, Any],
    promotion_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a failover plan for adversarial coverage gaps.
    
    Identifies metrics without adequate scenario coverage and determines
    if failover scenarios exist or if promotion should be blocked.
    
    Args:
        coverage_index: A coverage index from build_metric_adversarial_coverage_index()
        promotion_eval: A readiness eval from evaluate_adversarial_readiness_for_promotion()
    
    Returns:
        Dict with:
        - has_failover: bool (True if all critical metrics have coverage)
        - metrics_without_failover: list of metric names without adequate coverage
        - status: "OK" | "ATTENTION" | "BLOCK"
        - recommendations: list of neutral recommendation strings
    """
    metrics = coverage_index.get("metrics", {})
    global_data = coverage_index.get("global", {})
    metrics_at_risk = set(global_data.get("metrics_at_risk", []))
    
    metrics_blocking = set(promotion_eval.get("metrics_blocking_promotion", []))
    metrics_warning = set(promotion_eval.get("metrics_warning", []))
    
    metrics_without_failover = []
    recommendations = []
    
    # Check each metric for failover coverage
    for mk in METRIC_KINDS:
        metric_data = metrics.get(mk, {})
        coverage_status = metric_data.get("coverage_status", COVERAGE_STATUS["at_risk"])
        scenario_count = metric_data.get("scenario_count", 0)
        
        # Determine if metric has failover
        has_failover = False
        
        if scenario_count == 0:
            # No scenarios at all → no failover
            has_failover = False
        elif coverage_status == COVERAGE_STATUS["at_risk"]:
            # AT_RISK with scenarios → partial failover (regressions present)
            has_failover = scenario_count >= 1  # At least some coverage exists
        elif coverage_status == COVERAGE_STATUS["sparse"]:
            # SPARSE → limited failover
            has_failover = scenario_count >= 1
        else:
            # OK status → has failover
            has_failover = True
        
        if not has_failover:
            metrics_without_failover.append(mk)
            
            if mk in CORE_UPLIFT_METRICS:
                recommendations.append(f"{mk}: Core metric has no scenario coverage - blocking promotion")
            else:
                recommendations.append(f"{mk}: No scenario coverage - consider adding scenarios")
    
    # Determine overall status
    core_without_failover = [mk for mk in metrics_without_failover if mk in CORE_UPLIFT_METRICS]
    
    if len(core_without_failover) > 0:
        status = "BLOCK"
        has_failover_overall = False
    elif len(metrics_without_failover) > 0:
        status = "ATTENTION"
        has_failover_overall = False
    elif len(metrics_at_risk) > 0 or len(metrics_blocking) > 0:
        status = "ATTENTION"
        has_failover_overall = True  # Has coverage but issues present
    else:
        status = "OK"
        has_failover_overall = True
    
    return {
        "has_failover": has_failover_overall,
        "metrics_without_failover": sorted(metrics_without_failover),
        "status": status,
        "recommendations": sorted(recommendations),
    }


def curriculum_proposal_to_json(proposal: Dict[str, Any]) -> str:
    """Convert curriculum proposal to deterministic JSON string."""
    return json.dumps(proposal, indent=2, sort_keys=True)


def failover_plan_to_json(plan: Dict[str, Any]) -> str:
    """Convert failover plan to deterministic JSON string."""
    return json.dumps(plan, indent=2, sort_keys=True)


# ===========================================================================
# PHASE V — ADVERSARIAL-METRIC PRESSURE GRID
# ===========================================================================

# Pressure bands
PRESSURE_BANDS = {
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
}

# Pressure thresholds
PRESSURE_THRESHOLD_PRIORITY = 0.65  # Metrics with score > 0.65 become priority targets


def build_adversarial_pressure_model(
    coverage_index: Dict[str, Any],
    robustness_radar: Dict[str, Any],
    drift_grid: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build an adversarial pressure model combining coverage scarcity, fragility, and drift.
    
    Calculates pressure scores per metric that indicate how urgently new scenarios
    are needed. Combines multiple signals:
    - Coverage scarcity: How few scenarios exercise the metric
    - Adversarial fragility: Presence of regressions and cross-scenario issues
    - Drift volatility: Historical drift patterns (if provided)
    
    Args:
        coverage_index: A coverage index from build_metric_adversarial_coverage_index()
        robustness_radar: A radar from build_metric_robustness_radar()
        drift_grid: Optional dict with drift data per metric (default: None)
    
    Returns:
        Dict with:
        - metric_pressure_scores: Dict mapping metric -> pressure score (0.0-1.0)
        - scenario_pressure_targets: Dict mapping metric -> list of scenario names
        - global_pressure_index: Average pressure across all metrics
        - pressure_band: "LOW" | "MEDIUM" | "HIGH"
        - neutral_notes: List of neutral observation strings
    """
    metrics = coverage_index.get("metrics", {})
    radar_metrics = robustness_radar.get("metrics", {})
    global_data = robustness_radar.get("global", {})
    metrics_at_risk = set(global_data.get("metrics_at_risk", []))
    
    # Default drift_grid if not provided
    if drift_grid is None:
        drift_grid = {}
    
    metric_pressure_scores: Dict[str, float] = {}
    scenario_pressure_targets: Dict[str, List[str]] = {}
    notes = []
    
    for mk in METRIC_KINDS:
        metric_data = metrics.get(mk, {})
        radar_data = radar_metrics.get(mk, {})
        
        coverage_status = metric_data.get("coverage_status", COVERAGE_STATUS["at_risk"])
        scenario_count = metric_data.get("scenario_count", 0)
        regression_count = metric_data.get("regression_count", 0)
        scenarios_exercised = set(radar_data.get("scenarios_exercised", []))
        
        # Calculate pressure components (each 0.0-1.0)
        
        # 1. Coverage scarcity component (inverse of scenario count)
        if scenario_count == 0:
            scarcity_score = 1.0
        elif scenario_count == 1:
            scarcity_score = 0.7
        elif scenario_count == 2:
            scarcity_score = 0.4
        elif scenario_count >= 3:
            scarcity_score = 0.1
        else:
            scarcity_score = 0.5
        
        # 2. Adversarial fragility component
        if mk in metrics_at_risk:
            fragility_score = 1.0  # Cross-scenario regressions = maximum fragility
        elif regression_count > 0:
            fragility_score = 0.6 + min(0.3, regression_count / 10.0)  # Regressions present
        elif coverage_status == COVERAGE_STATUS["at_risk"]:
            fragility_score = 0.8  # AT_RISK even without regressions
        elif coverage_status == COVERAGE_STATUS["sparse"]:
            fragility_score = 0.4  # Sparse coverage = moderate fragility
        else:
            fragility_score = 0.1  # OK status = low fragility
        
        # 3. Drift volatility component (from drift_grid if available)
        drift_score = 0.0
        if mk in drift_grid:
            drift_data = drift_grid[mk]
            # Assume drift_grid has a "volatility" or "drift_count" field
            volatility = drift_data.get("volatility", 0.0)
            drift_count = drift_data.get("drift_count", 0)
            drift_score = min(1.0, volatility * 0.5 + drift_count * 0.1)
        
        # Combined pressure score (weighted average)
        # Coverage scarcity: 40%, Fragility: 40%, Drift: 20%
        pressure_score = (
            scarcity_score * 0.4 +
            fragility_score * 0.4 +
            drift_score * 0.2
        )
        
        metric_pressure_scores[mk] = round(pressure_score, 3)
        
        # Identify scenario pressure targets (score > threshold)
        if pressure_score > PRESSURE_THRESHOLD_PRIORITY:
            scenario_pressure_targets[mk] = sorted(list(scenarios_exercised))
            notes.append(f"{mk}: Pressure score {pressure_score:.3f} exceeds priority threshold")
    
    # Calculate global pressure index
    if metric_pressure_scores:
        global_pressure_index = sum(metric_pressure_scores.values()) / len(metric_pressure_scores)
    else:
        global_pressure_index = 0.0
    
    # Determine pressure band
    if global_pressure_index >= 0.7:
        pressure_band = PRESSURE_BANDS["high"]
    elif global_pressure_index >= 0.4:
        pressure_band = PRESSURE_BANDS["medium"]
    else:
        pressure_band = PRESSURE_BANDS["low"]
    
    return {
        "metric_pressure_scores": metric_pressure_scores,
        "scenario_pressure_targets": {k: sorted(v) for k, v in scenario_pressure_targets.items()},
        "global_pressure_index": round(global_pressure_index, 3),
        "pressure_band": pressure_band,
        "neutral_notes": sorted(notes),
    }


def build_evolving_adversarial_scenario_plan(
    pressure_model: Dict[str, Any],
    failover_plan: Dict[str, Any],
    promotion_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build an evolving multi-scenario plan based on pressure, failover, and promotion needs.
    
    Generates a prioritized backlog of scenarios to address coverage gaps,
    failover requirements, and promotion blockers.
    
    Args:
        pressure_model: A pressure model from build_adversarial_pressure_model()
        failover_plan: A failover plan from build_adversarial_failover_plan()
        promotion_eval: A readiness eval from evaluate_adversarial_readiness_for_promotion()
    
    Returns:
        Dict with:
        - scenario_backlog: List of scenario profile dicts to implement
        - priority_order: List of scenario names in priority order
        - multi_metric_scenarios: List of multi-metric scenario profiles
        - neutral_rationale: List of neutral explanation strings
    """
    pressure_scores = pressure_model.get("metric_pressure_scores", {})
    pressure_targets = pressure_model.get("scenario_pressure_targets", {})
    metrics_without_failover = set(failover_plan.get("metrics_without_failover", []))
    metrics_blocking = set(promotion_eval.get("metrics_blocking_promotion", []))
    metrics_warning = set(promotion_eval.get("metrics_warning", []))
    
    scenario_backlog = []
    priority_order = []
    multi_metric_scenarios = []
    rationale = []
    
    # Priority 1: Core metrics blocking promotion without failover
    core_blocking = [mk for mk in metrics_blocking if mk in CORE_UPLIFT_METRICS and mk in metrics_without_failover]
    for mk in sorted(core_blocking):
        scenario_name = f"{mk}_critical_failover"
        scenario_backlog.append({
            "name": scenario_name,
            "profile": "full",
            "metric_kinds": [mk],
            "modes": ["fault", "mutation", "replay"],
            "priority": 1,
            "rationale": f"Core metric {mk} blocking promotion with no failover coverage",
        })
        priority_order.append(scenario_name)
        rationale.append(f"Priority 1: {mk} requires immediate failover scenario")
    
    # Priority 2: High-pressure metrics (>0.65 threshold)
    high_pressure = [
        (mk, score) for mk, score in pressure_scores.items()
        if score > PRESSURE_THRESHOLD_PRIORITY and mk not in core_blocking
    ]
    high_pressure.sort(key=lambda x: -x[1])  # Sort by pressure descending
    
    for mk, score in high_pressure:
        scenario_name = f"{mk}_pressure_relief"
        scenario_backlog.append({
            "name": scenario_name,
            "profile": "full" if score > 0.8 else "standard",
            "metric_kinds": [mk],
            "modes": ["fault", "mutation", "replay"],
            "priority": 2,
            "rationale": f"High pressure score {score:.3f} for {mk}",
        })
        priority_order.append(scenario_name)
        rationale.append(f"Priority 2: {mk} pressure score {score:.3f} exceeds threshold")
    
    # Priority 3: Metrics without failover (non-core)
    non_core_no_failover = [mk for mk in metrics_without_failover if mk not in CORE_UPLIFT_METRICS]
    for mk in sorted(non_core_no_failover):
        if mk not in [s["metric_kinds"][0] for s in scenario_backlog]:
            scenario_name = f"{mk}_failover_coverage"
            scenario_backlog.append({
                "name": scenario_name,
                "profile": "standard",
                "metric_kinds": [mk],
                "modes": ["fault", "replay"],
                "priority": 3,
                "rationale": f"{mk} lacks failover coverage",
            })
            priority_order.append(scenario_name)
            rationale.append(f"Priority 3: {mk} requires failover scenario")
    
    # Priority 4: Warning metrics (sparse coverage)
    for mk in sorted(metrics_warning):
        if mk not in [s["metric_kinds"][0] for s in scenario_backlog]:
            scenario_name = f"{mk}_coverage_expansion"
            scenario_backlog.append({
                "name": scenario_name,
                "profile": "standard",
                "metric_kinds": [mk],
                "modes": ["fault", "mutation", "replay"],
                "priority": 4,
                "rationale": f"{mk} has sparse coverage",
            })
            priority_order.append(scenario_name)
            rationale.append(f"Priority 4: {mk} coverage expansion recommended")
    
    # Generate multi-metric scenarios for efficiency
    # Group metrics by priority and profile
    metrics_by_priority: Dict[int, List[str]] = {}
    for scenario in scenario_backlog:
        priority = scenario["priority"]
        mk = scenario["metric_kinds"][0]
        if priority not in metrics_by_priority:
            metrics_by_priority[priority] = []
        metrics_by_priority[priority].append(mk)
    
    # Create multi-metric scenarios for groups of 2-3 metrics
    for priority, metric_list in sorted(metrics_by_priority.items()):
        if len(metric_list) >= 2:
            # Group into chunks of 2-3
            for i in range(0, len(metric_list), 3):
                chunk = metric_list[i:i+3]
                if len(chunk) >= 2:
                    scenario_name = f"multi_metric_priority_{priority}_{len(multi_metric_scenarios) + 1}"
                    multi_metric_scenarios.append({
                        "name": scenario_name,
                        "profile": "standard",
                        "metric_kinds": sorted(chunk),
                        "modes": ["fault", "mutation", "replay"],
                        "priority": priority,
                        "rationale": f"Multi-metric scenario for {len(chunk)} metrics at priority {priority}",
                    })
                    rationale.append(f"Multi-metric scenario: {', '.join(chunk)} (priority {priority})")
    
    return {
        "scenario_backlog": sorted(scenario_backlog, key=lambda x: (x["priority"], x["name"])),
        "priority_order": priority_order,
        "multi_metric_scenarios": sorted(multi_metric_scenarios, key=lambda x: (x["priority"], x["name"])),
        "neutral_rationale": sorted(rationale),
    }


def build_adversarial_failover_plan_v2(
    coverage_index: Dict[str, Any],
    promotion_eval: Dict[str, Any],
    robustness_radar: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Enhanced failover plan with redundancy depth, scenario diversity, and failure-case sensitivity.
    
    Extends the base failover plan with additional metrics:
    - redundancy_depth: How many scenarios provide coverage (failover redundancy)
    - scenario_diversity: Variety of modes/profiles across scenarios
    - failure_case_sensitivity: How well scenarios cover failure cases
    
    Args:
        coverage_index: A coverage index from build_metric_adversarial_coverage_index()
        promotion_eval: A readiness eval from evaluate_adversarial_readiness_for_promotion()
        robustness_radar: Optional radar for diversity analysis (default: None)
    
    Returns:
        Enhanced failover plan with v2 metrics added
    """
    # Build base failover plan
    base_plan = build_adversarial_failover_plan(coverage_index, promotion_eval)
    
    metrics = coverage_index.get("metrics", {})
    radar_metrics = robustness_radar.get("metrics", {}) if robustness_radar else {}
    
    # Enhance with v2 metrics
    enhanced_metrics: Dict[str, Dict[str, Any]] = {}
    
    for mk in METRIC_KINDS:
        metric_data = metrics.get(mk, {})
        radar_data = radar_metrics.get(mk, {})
        
        scenario_count = metric_data.get("scenario_count", 0)
        scenarios_exercised = set(radar_data.get("scenarios_exercised", []))
        
        # Redundancy depth: number of scenarios providing coverage
        redundancy_depth = scenario_count
        
        # Scenario diversity: analyze modes and profiles across scenarios
        modes_seen = set()
        profiles_seen = set()
        
        for sc_name in scenarios_exercised:
            if sc_name in SCENARIOS:
                scenario = SCENARIOS[sc_name]
                modes_seen.update(scenario.modes)
                profiles_seen.add(scenario.profile)
        
        diversity_score = 0.0
        if len(modes_seen) >= 3:
            diversity_score += 0.5  # All modes covered
        elif len(modes_seen) >= 2:
            diversity_score += 0.3  # Most modes covered
        elif len(modes_seen) >= 1:
            diversity_score += 0.1  # Some modes covered
        
        if len(profiles_seen) >= 3:
            diversity_score += 0.5  # All profiles covered
        elif len(profiles_seen) >= 2:
            diversity_score += 0.3  # Multiple profiles
        elif len(profiles_seen) >= 1:
            diversity_score += 0.1  # Single profile
        
        scenario_diversity = min(1.0, diversity_score)
        
        # Failure-case sensitivity: how well scenarios cover failure modes
        regression_count = metric_data.get("regression_count", 0)
        coverage_status = metric_data.get("coverage_status", COVERAGE_STATUS["at_risk"])
        
        if regression_count > 0:
            # Has regressions detected → good sensitivity (caught issues)
            failure_sensitivity = 0.7 + min(0.3, regression_count / 10.0)
        elif coverage_status == COVERAGE_STATUS["at_risk"]:
            # AT_RISK but no regressions → moderate sensitivity
            failure_sensitivity = 0.5
        elif coverage_status == COVERAGE_STATUS["sparse"]:
            # Sparse → low sensitivity
            failure_sensitivity = 0.3
        else:
            # OK → high sensitivity (no issues found)
            failure_sensitivity = 0.9
        
        enhanced_metrics[mk] = {
            "redundancy_depth": redundancy_depth,
            "scenario_diversity": round(scenario_diversity, 3),
            "failure_case_sensitivity": round(failure_sensitivity, 3),
        }
    
    # Add v2 metrics to base plan
    base_plan["v2_metrics"] = enhanced_metrics
    
    # Calculate aggregate v2 scores
    if enhanced_metrics:
        avg_redundancy = sum(m["redundancy_depth"] for m in enhanced_metrics.values()) / len(enhanced_metrics)
        avg_diversity = sum(m["scenario_diversity"] for m in enhanced_metrics.values()) / len(enhanced_metrics)
        avg_sensitivity = sum(m["failure_case_sensitivity"] for m in enhanced_metrics.values()) / len(enhanced_metrics)
    else:
        avg_redundancy = 0.0
        avg_diversity = 0.0
        avg_sensitivity = 0.0
    
    base_plan["v2_aggregates"] = {
        "average_redundancy_depth": round(avg_redundancy, 2),
        "average_scenario_diversity": round(avg_diversity, 3),
        "average_failure_sensitivity": round(avg_sensitivity, 3),
    }
    
    return base_plan


def pressure_model_to_json(model: Dict[str, Any]) -> str:
    """Convert pressure model to deterministic JSON string."""
    return json.dumps(model, indent=2, sort_keys=True)


def scenario_plan_to_json(plan: Dict[str, Any]) -> str:
    """Convert scenario plan to deterministic JSON string."""
    return json.dumps(plan, indent=2, sort_keys=True)


# ===========================================================================
# REGRESSION RADAR SNAPSHOT GUARD
# ===========================================================================

@dataclass
class RadarSnapshot:
    """Snapshot of regression radar results for comparison."""
    schema_version: str
    seed: int
    timestamp: str
    results: List[Dict[str, Any]]
    summary_hash: str


def create_radar_snapshot(results: List[RegressionResult], seed: int = SEED_HARNESS) -> Dict[str, Any]:
    """
    Create a snapshot of radar results for persistence.
    
    The snapshot includes:
    - Schema version for compatibility
    - Seed used for determinism
    - Individual test results with hashes
    - Summary hash for quick comparison
    """
    import datetime
    
    results_data = []
    for r in results:
        results_data.append({
            "metric_kind": r.metric_kind,
            "test_name": r.test_name,
            "expected_hash": r.expected_hash,
            "actual_hash": r.actual_hash,
            "match": r.match,
        })
    
    # Create summary hash of all results
    summary_data = "|".join(
        f"{r['metric_kind']}:{r['test_name']}:{r['expected_hash']}"
        for r in sorted(results_data, key=lambda x: (x['metric_kind'], x['test_name']))
    )
    summary_hash = hashlib.sha256(summary_data.encode()).hexdigest()[:16]
    
    return {
        "schema_version": "1.0.0",
        "seed": seed,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "results": sorted(results_data, key=lambda x: (x['metric_kind'], x['test_name'])),
        "summary_hash": summary_hash,
    }


def snapshot_to_json(snapshot: Dict[str, Any]) -> str:
    """Convert snapshot to JSON for file storage."""
    return json.dumps(snapshot, indent=2, sort_keys=True)


def load_radar_snapshot(json_str: str) -> Dict[str, Any]:
    """Load a radar snapshot from JSON."""
    return json.loads(json_str)


def compare_radar_snapshots(
    current: Dict[str, Any],
    baseline: Dict[str, Any]
) -> Tuple[bool, List[Dict[str, str]]]:
    """
    Compare current radar results against a baseline snapshot.
    
    Returns:
        (match, list of differences)
        
    Each difference is a dict with:
        - metric_kind
        - test_name
        - baseline_hash
        - current_hash
    """
    differences = []
    
    # Build lookup from baseline
    baseline_lookup = {}
    for r in baseline.get("results", []):
        key = (r["metric_kind"], r["test_name"])
        baseline_lookup[key] = r
    
    # Compare current against baseline
    for r in current.get("results", []):
        key = (r["metric_kind"], r["test_name"])
        
        if key not in baseline_lookup:
            differences.append({
                "metric_kind": r["metric_kind"],
                "test_name": r["test_name"],
                "baseline_hash": "(missing)",
                "current_hash": r["expected_hash"],
                "reason": "new_test",
            })
        else:
            baseline_r = baseline_lookup[key]
            if r["expected_hash"] != baseline_r["expected_hash"]:
                differences.append({
                    "metric_kind": r["metric_kind"],
                    "test_name": r["test_name"],
                    "baseline_hash": baseline_r["expected_hash"],
                    "current_hash": r["expected_hash"],
                    "reason": "hash_changed",
                })
    
    # Check for removed tests
    current_keys = {(r["metric_kind"], r["test_name"]) for r in current.get("results", [])}
    for key, r in baseline_lookup.items():
        if key not in current_keys:
            differences.append({
                "metric_kind": r["metric_kind"],
                "test_name": r["test_name"],
                "baseline_hash": r["expected_hash"],
                "current_hash": "(removed)",
                "reason": "test_removed",
            })
    
    return len(differences) == 0, differences


def print_snapshot_diff(differences: List[Dict[str, str]]) -> None:
    """Print a concise diff of snapshot differences."""
    print("RADAR SNAPSHOT: MISMATCH")
    print("")
    
    for diff in sorted(differences, key=lambda x: (x["metric_kind"], x["test_name"])):
        print(f"  [{diff['metric_kind']}] {diff['test_name']}")
        print(f"    Baseline: {diff['baseline_hash']}")
        print(f"    Current:  {diff['current_hash']}")
        print(f"    Reason:   {diff['reason']}")
        print("")
    
    print(f"  {len(differences)} difference(s) detected")


# ===========================================================================
# OUTPUT FORMATTING
# ===========================================================================

def print_summary(summaries: List[HarnessSummary], verbose: bool = False) -> int:
    """Print a compact summary of results."""
    print("\n" + "=" * 60)
    print("METRIC TORTURE TEST — SUMMARY")
    print("=" * 60)
    
    total_cases = 0
    total_passed = 0
    total_mismatches = 0
    total_errors = 0
    
    for s in summaries:
        total_cases += s.total_cases
        total_passed += s.passed
        total_mismatches += s.mismatches
        total_errors += s.errors
        
        status = "✓ PASS" if (s.mismatches == 0 and s.errors < s.total_cases * 0.5) else "✗ FAIL"
        
        print(f"\n[{s.mode.upper()}] {s.metric_kind}")
        print(f"  Status:     {status}")
        print(f"  Total:      {s.total_cases}")
        print(f"  Passed:     {s.passed}")
        print(f"  Mismatches: {s.mismatches}")
        print(f"  Errors:     {s.errors}")
        print(f"  Duration:   {s.duration_seconds:.2f}s")
        
        if s.fault_divergences and verbose:
            print("  Divergences:")
            for fault, count in sorted(s.fault_divergences.items(), key=lambda x: -x[1]):
                print(f"    - {fault}: {count}")
    
    print("\n" + "-" * 60)
    print(f"TOTAL: {total_cases} cases, {total_passed} passed, {total_mismatches} mismatches, {total_errors} errors")
    
    if total_mismatches == 0 and total_errors < total_cases * 0.3:
        print("OVERALL: ✓ HARNESS PASSED")
        return 0
    else:
        print("OVERALL: ✗ HARNESS FAILED")
        return 1


# ===========================================================================
# PROFILE HELPERS
# ===========================================================================

def get_profile_test_count(profile: str) -> int:
    """Get the test count for a profile."""
    return PROFILE_CONFIG[profile]["n_tests"]


def get_profile_modes(profile: str) -> List[str]:
    """Get the modes for a profile."""
    return PROFILE_CONFIG[profile]["modes"]


# ===========================================================================
# CLI
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Metric Torture Test — Adversarial Harness CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile-based runs
  python -m experiments.metrics_adversarial_harness --profile fast --all-metrics
  python -m experiments.metrics_adversarial_harness --profile standard --metric-kind goal_hit
  python -m experiments.metrics_adversarial_harness --profile full --all-metrics  # Nightly
  
  # Scenario-based runs
  python -m experiments.metrics_adversarial_harness --scenario baseline_sanity
  python -m experiments.metrics_adversarial_harness --scenario multi_metric_stress
  python -m experiments.metrics_adversarial_harness --list-scenarios
  python -m experiments.metrics_adversarial_harness --list-scenarios --profile fast
  
  # Custom runs
  python -m experiments.metrics_adversarial_harness --mode fault --metric-kind goal_hit --n-tests 100
  python -m experiments.metrics_adversarial_harness --mode all --n-tests 500
  
  # Coverage report (JSON for CI)
  python -m experiments.metrics_adversarial_harness --coverage
  
  # Robustness scorecard (advisory)
  python -m experiments.metrics_adversarial_harness --scorecard --profile standard --all-metrics
  
  # Regression radar (pre-commit hook)
  python -m experiments.metrics_adversarial_harness --regression-radar
  python -m experiments.metrics_adversarial_harness --regression-radar --quiet
  
  # Radar snapshot guard
  python -m experiments.metrics_adversarial_harness --snapshot-radar radar_baseline.json
  python -m experiments.metrics_adversarial_harness --check-radar radar_baseline.json
  
  # Risk analytics (Phase III)
  python -m experiments.metrics_adversarial_harness --scenario-risk baseline_sanity --scorecard
  python -m experiments.metrics_adversarial_harness --robustness-radar
  python -m experiments.metrics_adversarial_harness --health-summary
  
  # Promotion gate & director panel (Phase IV)
  python -m experiments.metrics_adversarial_harness --coverage-index
  python -m experiments.metrics_adversarial_harness --promotion-readiness
  python -m experiments.metrics_adversarial_harness --director-panel
  
  # Curriculum designer & failover planner
  python -m experiments.metrics_adversarial_harness --propose-scenarios
  python -m experiments.metrics_adversarial_harness --failover-plan
  
  # Pressure grid & evolution plan (Phase V)
  python -m experiments.metrics_adversarial_harness --pressure-model
  python -m experiments.metrics_adversarial_harness --scenario-evolution-plan
  python -m experiments.metrics_adversarial_harness --failover-plan-v2
  
  # Profile contracts (JSON)
  python -m experiments.metrics_adversarial_harness --profile-contracts
        """
    )
    
    parser.add_argument(
        "--metric-kind",
        choices=METRIC_KINDS,
        default="goal_hit",
        help="Metric kind to test (default: goal_hit)"
    )
    
    parser.add_argument(
        "--n-tests",
        type=int,
        default=None,
        help="Number of test cases (overrides profile)"
    )
    
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=None,
        help="Test mode: fault, mutation, replay, or all"
    )
    
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default=None,
        help="Test profile: fast (~50), standard (~250), full (~1000+)"
    )
    
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="Run tests for all metric kinds"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output with divergence details"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet output (minimal, for CI)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED_HARNESS,
        help=f"Random seed for reproducibility (default: {SEED_HARNESS})"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Output JSON coverage report"
    )
    
    parser.add_argument(
        "--regression-radar",
        action="store_true",
        help="Run regression radar (fixed battery, compare against expected)"
    )
    
    parser.add_argument(
        "--profile-contracts",
        action="store_true",
        help="Output profile contracts as JSON"
    )
    
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default=None,
        help="Run a named adversarial scenario"
    )
    
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios (optionally filter by --profile)"
    )
    
    parser.add_argument(
        "--scorecard",
        action="store_true",
        help="Generate robustness scorecard (advisory, JSON)"
    )
    
    parser.add_argument(
        "--snapshot-radar",
        type=str,
        metavar="PATH",
        default=None,
        help="Save radar snapshot to file"
    )
    
    parser.add_argument(
        "--check-radar",
        type=str,
        metavar="PATH",
        default=None,
        help="Check radar against snapshot file (exit 1 on mismatch)"
    )
    
    parser.add_argument(
        "--scenario-risk",
        type=str,
        choices=list(SCENARIOS.keys()),
        default=None,
        metavar="SCENARIO",
        help="Generate risk summary for a specific scenario (requires --scorecard data)"
    )
    
    parser.add_argument(
        "--robustness-radar",
        action="store_true",
        help="Build multi-scenario robustness radar (JSON)"
    )
    
    parser.add_argument(
        "--health-summary",
        action="store_true",
        help="Generate adversarial health summary for dashboards"
    )
    
    parser.add_argument(
        "--coverage-index",
        action="store_true",
        help="Build metric adversarial coverage index (Phase IV)"
    )
    
    parser.add_argument(
        "--promotion-readiness",
        action="store_true",
        help="Evaluate promotion readiness from adversarial view (Phase IV)"
    )
    
    parser.add_argument(
        "--director-panel",
        action="store_true",
        help="Build adversarial director panel (Phase IV)"
    )
    
    parser.add_argument(
        "--propose-scenarios",
        action="store_true",
        help="Propose new adversarial scenarios based on coverage gaps"
    )
    
    parser.add_argument(
        "--failover-plan",
        action="store_true",
        help="Build adversarial failover plan for coverage gaps"
    )
    
    parser.add_argument(
        "--pressure-model",
        action="store_true",
        help="Build adversarial pressure model (Phase V)"
    )
    
    parser.add_argument(
        "--scenario-evolution-plan",
        action="store_true",
        help="Build evolving adversarial scenario plan (Phase V)"
    )
    
    parser.add_argument(
        "--failover-plan-v2",
        action="store_true",
        help="Build enhanced failover plan v2 with redundancy/diversity metrics"
    )
    
    args = parser.parse_args()
    
    # Handle profile contracts mode
    if args.profile_contracts:
        print(get_profile_contract_json())
        return 0
    
    # Handle list scenarios mode
    if args.list_scenarios:
        scenarios = list_scenarios(filter_profile=args.profile)
        print(json.dumps([{
            "name": s.name,
            "profile": s.profile,
            "metric_kinds": list(s.metric_kinds),
            "modes": list(s.modes),
            "description": s.description,
        } for s in scenarios], indent=2))
        return 0
    
    # Handle coverage mode
    if args.coverage:
        report = generate_coverage_report()
        print(coverage_report_to_json(report))
        return 0
    
    # Handle snapshot radar mode
    if args.snapshot_radar:
        results = run_regression_radar(args.seed)
        snapshot = create_radar_snapshot(results, seed=args.seed)
        snapshot_json = snapshot_to_json(snapshot)
        
        with open(args.snapshot_radar, 'w') as f:
            f.write(snapshot_json)
        
        print(f"RADAR SNAPSHOT: Saved to {args.snapshot_radar}")
        print(f"  Summary hash: {snapshot['summary_hash']}")
        print(f"  {len(results)} tests captured")
        return 0
    
    # Handle check radar mode
    if args.check_radar:
        # Load baseline
        try:
            with open(args.check_radar, 'r') as f:
                baseline = load_radar_snapshot(f.read())
        except FileNotFoundError:
            print(f"ERROR: Snapshot file not found: {args.check_radar}")
            return 1
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid snapshot JSON: {e}")
            return 1
        
        # Run current radar
        results = run_regression_radar(args.seed)
        current = create_radar_snapshot(results, seed=args.seed)
        
        # Compare
        match, differences = compare_radar_snapshots(current, baseline)
        
        if match:
            print("RADAR SNAPSHOT: OK")
            print(f"  Summary hash: {current['summary_hash']}")
            return 0
        else:
            print_snapshot_diff(differences)
            return 1
    
    # Handle regression radar mode
    if args.regression_radar:
        results = run_regression_radar(args.seed)
        return print_regression_radar_results(results, quiet=args.quiet)
    
    # Handle robustness radar mode (multi-scenario analysis)
    if args.robustness_radar:
        # Run all scenarios and collect scorecards
        harness = AdversarialHarness(seed=args.seed)
        scorecards = []
        
        for scenario_name, scenario in sorted(SCENARIOS.items()):
            all_summaries = []
            n_tests = PROFILE_CONFIG[scenario.profile]["n_tests"]
            
            for mk in scenario.metric_kinds:
                for mode in scenario.modes:
                    summaries = harness.run(mk, n_tests, mode)
                    all_summaries.extend(summaries)
            
            scorecard = build_robustness_scorecard(all_summaries, scenarios_run=[scenario_name])
            scorecards.append(scorecard)
        
        radar = build_metric_robustness_radar(scorecards)
        print(json.dumps(radar, indent=2, sort_keys=True))
        return 0
    
    # Handle health summary mode
    if args.health_summary:
        # Build radar first (using fast scenarios for speed)
        harness = AdversarialHarness(seed=args.seed)
        scorecards = []
        
        # Use only fast scenarios for quick health check
        fast_scenarios = list_scenarios(filter_profile="fast")
        
        for scenario in fast_scenarios:
            all_summaries = []
            n_tests = PROFILE_CONFIG[scenario.profile]["n_tests"]
            
            for mk in scenario.metric_kinds:
                for mode in scenario.modes:
                    summaries = harness.run(mk, n_tests, mode)
                    all_summaries.extend(summaries)
            
            scorecard = build_robustness_scorecard(all_summaries, scenarios_run=[scenario.name])
            scorecards.append(scorecard)
        
        radar = build_metric_robustness_radar(scorecards)
        health = summarize_adversarial_health_for_global_health(radar)
        print(health_summary_to_json(health))
        return 0 if health["adversarial_coverage_ok"] else 1
    
    # Handle coverage index mode (Phase IV)
    if args.coverage_index:
        # Build radar first
        harness = AdversarialHarness(seed=args.seed)
        scorecards = []
        
        # Use fast scenarios for quick check
        fast_scenarios = list_scenarios(filter_profile="fast")
        
        for scenario in fast_scenarios:
            all_summaries = []
            n_tests = PROFILE_CONFIG[scenario.profile]["n_tests"]
            
            for mk in scenario.metric_kinds:
                for mode in scenario.modes:
                    summaries = harness.run(mk, n_tests, mode)
                    all_summaries.extend(summaries)
            
            scorecard = build_robustness_scorecard(all_summaries, scenarios_run=[scenario.name])
            scorecards.append(scorecard)
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        print(coverage_index_to_json(coverage_index))
        return 0 if coverage_index["global"]["coverage_ok"] else 1
    
    # Handle promotion readiness mode (Phase IV)
    if args.promotion_readiness:
        # Build radar and coverage index
        harness = AdversarialHarness(seed=args.seed)
        scorecards = []
        
        fast_scenarios = list_scenarios(filter_profile="fast")
        
        for scenario in fast_scenarios:
            all_summaries = []
            n_tests = PROFILE_CONFIG[scenario.profile]["n_tests"]
            
            for mk in scenario.metric_kinds:
                for mode in scenario.modes:
                    summaries = harness.run(mk, n_tests, mode)
                    all_summaries.extend(summaries)
            
            scorecard = build_robustness_scorecard(all_summaries, scenarios_run=[scenario.name])
            scorecards.append(scorecard)
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        print(readiness_eval_to_json(readiness))
        return 0 if readiness["promotion_ok"] else 1
    
    # Handle director panel mode (Phase IV)
    if args.director_panel:
        # Build radar, coverage index, and readiness
        harness = AdversarialHarness(seed=args.seed)
        scorecards = []
        
        fast_scenarios = list_scenarios(filter_profile="fast")
        
        for scenario in fast_scenarios:
            all_summaries = []
            n_tests = PROFILE_CONFIG[scenario.profile]["n_tests"]
            
            for mk in scenario.metric_kinds:
                for mode in scenario.modes:
                    summaries = harness.run(mk, n_tests, mode)
                    all_summaries.extend(summaries)
            
            scorecard = build_robustness_scorecard(all_summaries, scenarios_run=[scenario.name])
            scorecards.append(scorecard)
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        panel = build_adversarial_director_panel(coverage_index, readiness)
        print(director_panel_to_json(panel))
        return 0 if panel["adversarial_coverage_ok"] and readiness["promotion_ok"] else 1
    
    # Handle propose scenarios mode
    if args.propose_scenarios:
        # Build radar and coverage index
        harness = AdversarialHarness(seed=args.seed)
        scorecards = []
        
        fast_scenarios = list_scenarios(filter_profile="fast")
        
        for scenario in fast_scenarios:
            all_summaries = []
            n_tests = PROFILE_CONFIG[scenario.profile]["n_tests"]
            
            for mk in scenario.metric_kinds:
                for mode in scenario.modes:
                    summaries = harness.run(mk, n_tests, mode)
                    all_summaries.extend(summaries)
            
            scorecard = build_robustness_scorecard(all_summaries, scenarios_run=[scenario.name])
            scorecards.append(scorecard)
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        proposal = propose_adversarial_scenarios(coverage_index, radar)
        print(curriculum_proposal_to_json(proposal))
        return 0
    
    # Handle failover plan mode
    if args.failover_plan:
        # Build radar, coverage index, and readiness
        harness = AdversarialHarness(seed=args.seed)
        scorecards = []
        
        fast_scenarios = list_scenarios(filter_profile="fast")
        
        for scenario in fast_scenarios:
            all_summaries = []
            n_tests = PROFILE_CONFIG[scenario.profile]["n_tests"]
            
            for mk in scenario.metric_kinds:
                for mode in scenario.modes:
                    summaries = harness.run(mk, n_tests, mode)
                    all_summaries.extend(summaries)
            
            scorecard = build_robustness_scorecard(all_summaries, scenarios_run=[scenario.name])
            scorecards.append(scorecard)
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        failover_plan = build_adversarial_failover_plan(coverage_index, readiness)
        print(failover_plan_to_json(failover_plan))
        return 0 if failover_plan["has_failover"] else 1
    
    # Handle pressure model mode (Phase V)
    if args.pressure_model:
        # Build radar and coverage index
        harness = AdversarialHarness(seed=args.seed)
        scorecards = []
        
        fast_scenarios = list_scenarios(filter_profile="fast")
        
        for scenario in fast_scenarios:
            all_summaries = []
            n_tests = PROFILE_CONFIG[scenario.profile]["n_tests"]
            
            for mk in scenario.metric_kinds:
                for mode in scenario.modes:
                    summaries = harness.run(mk, n_tests, mode)
                    all_summaries.extend(summaries)
            
            scorecard = build_robustness_scorecard(all_summaries, scenarios_run=[scenario.name])
            scorecards.append(scorecard)
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        pressure_model = build_adversarial_pressure_model(coverage_index, radar)
        print(pressure_model_to_json(pressure_model))
        return 0
    
    # Handle scenario evolution plan mode (Phase V)
    if args.scenario_evolution_plan:
        # Build all required components
        harness = AdversarialHarness(seed=args.seed)
        scorecards = []
        
        fast_scenarios = list_scenarios(filter_profile="fast")
        
        for scenario in fast_scenarios:
            all_summaries = []
            n_tests = PROFILE_CONFIG[scenario.profile]["n_tests"]
            
            for mk in scenario.metric_kinds:
                for mode in scenario.modes:
                    summaries = harness.run(mk, n_tests, mode)
                    all_summaries.extend(summaries)
            
            scorecard = build_robustness_scorecard(all_summaries, scenarios_run=[scenario.name])
            scorecards.append(scorecard)
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        failover_plan = build_adversarial_failover_plan(coverage_index, readiness)
        pressure_model = build_adversarial_pressure_model(coverage_index, radar)
        evolution_plan = build_evolving_adversarial_scenario_plan(pressure_model, failover_plan, readiness)
        print(scenario_plan_to_json(evolution_plan))
        return 0
    
    # Handle failover plan v2 mode (Phase V)
    if args.failover_plan_v2:
        # Build radar, coverage index, and readiness
        harness = AdversarialHarness(seed=args.seed)
        scorecards = []
        
        fast_scenarios = list_scenarios(filter_profile="fast")
        
        for scenario in fast_scenarios:
            all_summaries = []
            n_tests = PROFILE_CONFIG[scenario.profile]["n_tests"]
            
            for mk in scenario.metric_kinds:
                for mode in scenario.modes:
                    summaries = harness.run(mk, n_tests, mode)
                    all_summaries.extend(summaries)
            
            scorecard = build_robustness_scorecard(all_summaries, scenarios_run=[scenario.name])
            scorecards.append(scorecard)
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        failover_plan_v2 = build_adversarial_failover_plan_v2(coverage_index, readiness, radar)
        print(failover_plan_to_json(failover_plan_v2))
        return 0 if failover_plan_v2["has_failover"] else 1
    
    # Handle scenario mode
    if args.scenario:
        scenario = get_scenario(args.scenario)
        harness = AdversarialHarness(seed=args.seed)
        all_summaries: List[HarnessSummary] = []
        
        n_tests = PROFILE_CONFIG[scenario.profile]["n_tests"]
        
        print(f"\nRunning Scenario: {scenario.name}")
        print(f"  Profile: {scenario.profile}")
        print(f"  Metrics: {', '.join(scenario.metric_kinds)}")
        print(f"  Modes: {', '.join(scenario.modes)}")
        print(f"  Tests per mode: {n_tests}")
        print(f"  Seed: {args.seed}")
        
        for mk in scenario.metric_kinds:
            for mode in scenario.modes:
                summaries = harness.run(mk, n_tests, mode)
                all_summaries.extend(summaries)
        
        if args.scorecard or args.scenario_risk:
            scorecard = build_robustness_scorecard(all_summaries, scenarios_run=[args.scenario])
            
            if args.scenario_risk:
                # Generate scenario risk summary
                risk = summarize_scenario_risk(scorecard, args.scenario_risk)
                print(json.dumps(risk, indent=2, sort_keys=True))
                return 0
            else:
                print("\n" + scorecard_to_json(scorecard))
                return 0
        
        return print_summary(all_summaries, verbose=args.verbose)
    
    # Determine test parameters
    if args.profile:
        n_tests = PROFILE_CONFIG[args.profile]["n_tests"]
        mode = "all"
    else:
        n_tests = args.n_tests or 100
        mode = args.mode or "all"
    
    harness = AdversarialHarness(seed=args.seed)
    all_summaries: List[HarnessSummary] = []
    
    metric_kinds = METRIC_KINDS if args.all_metrics else [args.metric_kind]
    
    profile_info = f" (profile: {args.profile})" if args.profile else ""
    print(f"\nRunning Metric Torture Test{profile_info}...")
    print(f"  Mode: {mode}")
    print(f"  Metric(s): {', '.join(metric_kinds)}")
    print(f"  Tests per mode: {n_tests}")
    print(f"  Seed: {args.seed}")
    
    for mk in metric_kinds:
        if args.profile:
            summaries = harness.run_with_profile(mk, args.profile)
        else:
            summaries = harness.run(mk, n_tests, mode)
        all_summaries.extend(summaries)
    
    # Output scorecard if requested
    if args.scorecard:
        scorecard = build_robustness_scorecard(all_summaries)
        print(scorecard_to_json(scorecard))
        return 0
    
    return print_summary(all_summaries, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
