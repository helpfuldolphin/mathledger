"""
Pytest fixtures for Mock Oracle tests.

Provides common fixtures for mock oracle testing, ensuring consistent
configuration and deterministic behavior across all test modules.
"""

from __future__ import annotations

import os
import pytest

# Enable mock oracle for all tests in this package
os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"

from backend.verification import (
    MockOracleConfig,
    MockVerifiableOracle,
    MockOracleExpectations,
    SLICE_PROFILES,
)


@pytest.fixture
def default_config() -> MockOracleConfig:
    """Default mock oracle configuration."""
    return MockOracleConfig(
        slice_profile="default",
        timeout_ms=50,
        enable_crashes=False,
        latency_jitter_pct=0.10,
        seed=42,
    )


@pytest.fixture
def default_oracle(default_config: MockOracleConfig) -> MockVerifiableOracle:
    """Default mock oracle instance."""
    return MockVerifiableOracle(default_config)


@pytest.fixture
def crash_enabled_config() -> MockOracleConfig:
    """Configuration with crashes enabled."""
    return MockOracleConfig(
        slice_profile="default",
        timeout_ms=50,
        enable_crashes=True,
        latency_jitter_pct=0.10,
        seed=42,
    )


@pytest.fixture
def crash_enabled_oracle(crash_enabled_config: MockOracleConfig) -> MockVerifiableOracle:
    """Mock oracle with crashes enabled."""
    return MockVerifiableOracle(crash_enabled_config)


@pytest.fixture
def goal_hit_oracle() -> MockVerifiableOracle:
    """Mock oracle configured for goal_hit profile."""
    config = MockOracleConfig(slice_profile="goal_hit", seed=42)
    return MockVerifiableOracle(config)


@pytest.fixture
def sparse_oracle() -> MockVerifiableOracle:
    """Mock oracle configured for sparse profile."""
    config = MockOracleConfig(slice_profile="sparse", seed=42)
    return MockVerifiableOracle(config)


@pytest.fixture
def tree_oracle() -> MockVerifiableOracle:
    """Mock oracle configured for tree profile."""
    config = MockOracleConfig(slice_profile="tree", seed=42)
    return MockVerifiableOracle(config)


@pytest.fixture
def dependency_oracle() -> MockVerifiableOracle:
    """Mock oracle configured for dependency profile."""
    config = MockOracleConfig(slice_profile="dependency", seed=42)
    return MockVerifiableOracle(config)


@pytest.fixture
def sample_formulas() -> list[str]:
    """Collection of sample formulas for testing."""
    return [
        "p -> p",
        "p -> (q -> p)",
        "(p -> (q -> r)) -> ((p -> q) -> (p -> r))",
        "p /\\ q -> p",
        "p -> p \\/ q",
        "~(p /\\ ~p)",
        "p \\/ ~p",
        "(p -> q) -> (~q -> ~p)",
        "((p -> q) /\\ (q -> r)) -> (p -> r)",
        "~(p /\\ q) -> (~p \\/ ~q)",
    ]


@pytest.fixture
def many_formulas() -> list[str]:
    """Large collection of formulas for distribution testing."""
    atoms = ["p", "q", "r", "s", "t"]
    ops = ["->", "/\\", "\\/"]
    formulas = []
    
    for i in range(1000):
        a1 = atoms[i % len(atoms)]
        a2 = atoms[(i // len(atoms)) % len(atoms)]
        op = ops[(i // (len(atoms) ** 2)) % len(ops)]
        formulas.append(f"{a1} {op} {a2}_{i}")
    
    return formulas


@pytest.fixture
def all_profiles() -> list[str]:
    """List of all available profiles."""
    return list(SLICE_PROFILES.keys())

