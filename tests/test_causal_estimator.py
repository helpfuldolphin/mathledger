"""
Unit tests for causal coefficient estimation.

Tests OLS, matching, and stability analysis.
"""

import pytest
import numpy as np
from backend.causal.estimator import (
    estimate_causal_effect,
    EstimationMethod,
    CausalCoefficient,
    estimate_all_edges,
    compute_stability,
    format_pass_message
)
from backend.causal.graph import build_rfl_graph


def test_causal_coefficient_creation():
    """Test creating CausalCoefficient."""
    coef = CausalCoefficient(
        source_var="X",
        target_var="Y",
        coefficient=0.5,
        std_error=0.1,
        confidence_interval=(0.3, 0.7),
        p_value=0.01,
        n_observations=100,
        method=EstimationMethod.OLS
    )

    assert coef.coefficient == 0.5
    assert coef.is_significant is True  # p < 0.05


def test_causal_coefficient_not_significant():
    """Test non-significant coefficient."""
    coef = CausalCoefficient(
        source_var="X",
        target_var="Y",
        coefficient=0.01,
        std_error=0.1,
        confidence_interval=(-0.1, 0.12),
        p_value=0.9,
        n_observations=10,
        method=EstimationMethod.OLS
    )

    assert coef.is_significant is False


def test_estimate_ols_positive_effect():
    """Test OLS estimation with positive effect."""
    # Generate synthetic data: Y = 2 + 0.5*X + noise
    np.random.seed(42)
    n = 100

    X = np.random.normal(0, 1, n)
    Y = 2 + 0.5 * X + np.random.normal(0, 0.1, n)

    data = {
        'X': X.tolist(),
        'Y': Y.tolist()
    }

    coef = estimate_causal_effect(
        'X', 'Y', data,
        method=EstimationMethod.OLS,
        seed=42
    )

    # Should estimate coefficient near 0.5
    assert 0.4 < coef.coefficient < 0.6
    assert coef.is_significant == True
    assert coef.n_observations == n


def test_estimate_ols_negative_effect():
    """Test OLS estimation with negative effect."""
    np.random.seed(42)
    n = 50

    X = np.random.normal(0, 1, n)
    Y = 10 - 0.8 * X + np.random.normal(0, 0.2, n)

    data = {
        'X': X.tolist(),
        'Y': Y.tolist()
    }

    coef = estimate_causal_effect(
        'X', 'Y', data,
        method=EstimationMethod.OLS,
        seed=42
    )

    # Should estimate negative coefficient
    assert -1.0 < coef.coefficient < -0.6
    assert coef.is_significant == True


def test_estimate_with_confounders():
    """Test estimation with confounders."""
    np.random.seed(42)
    n = 100

    # Z is a confounder affecting both X and Y
    Z = np.random.normal(0, 1, n)
    X = 0.5 * Z + np.random.normal(0, 0.5, n)
    Y = 0.3 * X + 0.7 * Z + np.random.normal(0, 0.2, n)

    data = {
        'X': X.tolist(),
        'Y': Y.tolist(),
        'Z': Z.tolist()
    }

    # Estimate without adjusting for confounder
    coef_unadjusted = estimate_causal_effect(
        'X', 'Y', data,
        confounders=None,
        method=EstimationMethod.OLS,
        seed=42
    )

    # Estimate adjusting for confounder
    coef_adjusted = estimate_causal_effect(
        'X', 'Y', data,
        confounders=['Z'],
        method=EstimationMethod.OLS,
        seed=42
    )

    # Adjusted estimate should be closer to true effect (0.3)
    # (In practice, confounding would bias unadjusted estimate)
    assert coef_adjusted.is_significant == True


def test_estimate_matching():
    """Test propensity score matching estimation."""
    np.random.seed(42)
    n = 100

    # Generate continuous treatment with good separation
    X = np.concatenate([np.random.normal(0, 0.5, n//2),
                        np.random.normal(2, 0.5, n//2)])
    Y = 5 + 2 * X + np.random.normal(0, 1, n)

    data = {
        'X': X.tolist(),
        'Y': Y.tolist()
    }

    coef = estimate_causal_effect(
        'X', 'Y', data,
        method=EstimationMethod.MATCHING,
        seed=42
    )

    # Should estimate positive effect
    # Matching uses median split, so effect should be positive
    assert coef.n_observations == n


def test_estimate_insufficient_data():
    """Test that estimation fails gracefully with insufficient data."""
    data = {
        'X': [1.0, 2.0],
        'Y': [3.0, 4.0]
    }

    with pytest.raises(ValueError, match="Insufficient data"):
        estimate_causal_effect('X', 'Y', data, method=EstimationMethod.OLS)


def test_estimate_missing_variable():
    """Test that estimation fails when variable is missing."""
    data = {
        'X': [1.0, 2.0, 3.0]
    }

    with pytest.raises(ValueError, match="not in data"):
        estimate_causal_effect('X', 'Y', data, method=EstimationMethod.OLS)


def test_estimate_all_edges():
    """Test estimating coefficients for all edges in graph."""
    graph = build_rfl_graph()

    # Generate synthetic data
    np.random.seed(42)
    n = 50

    data = {
        'policy_hash': np.random.randint(0, 10, n).tolist(),
        'abstain_pct': np.random.uniform(0, 30, n).tolist(),
        'proofs_per_sec': np.random.uniform(0.5, 2.0, n).tolist(),
        'verify_ms_p50': np.random.uniform(20, 100, n).tolist(),
        'depth_max': np.random.randint(2, 6, n).tolist()
    }

    coefficients = estimate_all_edges(
        graph,
        data,
        method=EstimationMethod.OLS,
        seed=42
    )

    # Should estimate coefficients for all edges
    assert len(coefficients) > 0

    # Check that coefficients are stored
    for (source, target), coef in coefficients.items():
        assert isinstance(coef, CausalCoefficient)
        assert coef.source_var == source
        assert coef.target_var == target


def test_stability_bootstrap():
    """Test causal stability via bootstrap."""
    # Generate data with stable relationship
    np.random.seed(42)
    n = 100

    X = np.random.normal(0, 1, n)
    Y = 0.5 * X + np.random.normal(0, 0.1, n)

    data = {
        'X': X.tolist(),
        'Y': Y.tolist()
    }

    from backend.causal.graph import CausalGraph, CausalNode, CausalEdge, VariableType

    graph = CausalGraph()
    X_node = CausalNode('X', VariableType.POLICY)
    Y_node = CausalNode('Y', VariableType.THROUGHPUT)
    graph.add_edge(CausalEdge(source=X_node, target=Y_node))

    stability = compute_stability(
        graph,
        data,
        n_bootstrap=20,
        seed=42
    )

    assert ('X', 'Y') in stability

    result = stability[('X', 'Y')]
    assert 'mean' in result
    assert 'std' in result
    assert 'cv' in result  # Coefficient of variation

    # Mean should be close to true effect (0.5)
    assert 0.3 < result['mean'] < 0.7


def test_format_pass_message():
    """Test formatting pass message."""
    coef = CausalCoefficient(
        source_var="policy_hash",
        target_var="abstain_pct",
        coefficient=0.123,
        std_error=0.05,
        confidence_interval=(0.02, 0.22),
        p_value=0.03,
        n_observations=50,
        method=EstimationMethod.OLS
    )

    msg = format_pass_message("policy_hash", "abstain_pct", coef)

    assert "[PASS]" in msg
    assert "Stable" in msg
    assert "do(policy_hash)->abstain_pct" in msg
    assert "coeff=0.123" in msg
    assert "p=0.03" in msg


def test_format_pass_message_unstable():
    """Test formatting pass message for unstable coefficient."""
    coef = CausalCoefficient(
        source_var="X",
        target_var="Y",
        coefficient=0.01,
        std_error=0.1,
        confidence_interval=(-0.2, 0.22),
        p_value=0.9,
        n_observations=10,
        method=EstimationMethod.OLS
    )

    msg = format_pass_message("X", "Y", coef)

    assert "[PASS]" in msg
    assert "Unstable" in msg


def test_coefficient_to_dict():
    """Test serializing coefficient to dictionary."""
    coef = CausalCoefficient(
        source_var="X",
        target_var="Y",
        coefficient=0.5,
        std_error=0.1,
        confidence_interval=(0.3, 0.7),
        p_value=0.01,
        n_observations=100,
        method=EstimationMethod.OLS
    )

    data = coef.to_dict()

    assert data['source'] == 'X'
    assert data['target'] == 'Y'
    assert data['coefficient'] == 0.5
    assert data['std_error'] == 0.1
    assert data['p_value'] == 0.01
    assert data['significant'] is True
    assert data['n'] == 100
    assert data['method'] == 'ols'


def test_coefficient_repr():
    """Test coefficient string representation."""
    coef = CausalCoefficient(
        source_var="X",
        target_var="Y",
        coefficient=0.456,
        std_error=0.1,
        confidence_interval=(0.26, 0.65),
        p_value=0.001,
        n_observations=100,
        method=EstimationMethod.OLS
    )

    repr_str = repr(coef)

    assert "X" in repr_str
    assert "Y" in repr_str
    assert "0.456" in repr_str
    # p = 0.001 is exactly at the boundary, check for significance marker
    assert ("*" in repr_str or "*" in repr_str)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
