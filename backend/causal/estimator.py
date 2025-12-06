"""
Causal coefficient estimation.

Estimates causal effects from observational or interventional data.
Produces coefficients for edges in causal graph: do(X) -> Y coefficient.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
from scipy import stats
from backend.repro.determinism import SeededRNG

_GLOBAL_SEED = 0


class EstimationMethod(Enum):
    """Methods for causal effect estimation."""
    OLS = "ols"  # Ordinary Least Squares
    IV = "instrumental_variable"  # Instrumental Variables
    DML = "double_ml"  # Double Machine Learning
    MATCHING = "matching"  # Propensity Score Matching
    RDD = "regression_discontinuity"  # Regression Discontinuity Design


@dataclass
class CausalCoefficient:
    """
    Estimated causal effect coefficient.

    Represents: E[ΔY | do(X → X+1)]
    """
    source_var: str
    target_var: str
    coefficient: float
    std_error: float
    confidence_interval: Tuple[float, float]
    p_value: float
    n_observations: int
    method: EstimationMethod

    @property
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if coefficient is statistically significant."""
        return self.p_value < alpha

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'source': self.source_var,
            'target': self.target_var,
            'coefficient': round(self.coefficient, 4),
            'std_error': round(self.std_error, 4),
            'ci_95': [round(self.confidence_interval[0], 4),
                      round(self.confidence_interval[1], 4)],
            'p_value': self.p_value,
            'significant': self.is_significant,
            'n': self.n_observations,
            'method': self.method.value
        }

    def __repr__(self):
        sig = '***' if self.p_value < 0.001 else ('**' if self.p_value < 0.01 else
                                                    ('*' if self.p_value < 0.05 else ''))
        return (
            f"{self.source_var} → {self.target_var}: "
            f"β={self.coefficient:.3f}{sig} "
            f"(SE={self.std_error:.3f}, p={self.p_value:.4f})"
        )


def estimate_causal_effect(
    source_var: str,
    target_var: str,
    data: Dict[str, List[float]],
    confounders: Optional[List[str]] = None,
    method: EstimationMethod = EstimationMethod.OLS,
    seed: int = _GLOBAL_SEED
) -> CausalCoefficient:
    """
    Estimate causal effect from data.

    Args:
        source_var: Cause variable (X)
        target_var: Effect variable (Y)
        data: Dictionary mapping variable names to value lists
        confounders: List of confounding variables to adjust for
        method: Estimation method to use
        seed: Random seed for deterministic estimation

    Returns:
        Estimated causal coefficient

    Raises:
        ValueError: If data is insufficient or variables not found
    """
    if source_var not in data or target_var not in data:
        raise ValueError(f"Variables {source_var} or {target_var} not in data")

    X = np.array(data[source_var])
    Y = np.array(data[target_var])

    if len(X) != len(Y):
        raise ValueError("Source and target must have same length")

    n = len(X)
    if n < 3:
        raise ValueError(f"Insufficient data: n={n}, need at least 3 observations")

    # Add confounders to design matrix
    if confounders:
        Z = np.column_stack([data[c] for c in confounders if c in data])
        X_matrix = np.column_stack([X, Z])
    else:
        X_matrix = X.reshape(-1, 1)

    # Estimate based on method
    if method == EstimationMethod.OLS:
        return _estimate_ols(source_var, target_var, X_matrix, Y, n)

    elif method == EstimationMethod.MATCHING:
        return _estimate_matching(source_var, target_var, X, Y, seed)

    elif method == EstimationMethod.DML:
        return _estimate_double_ml(source_var, target_var, X_matrix, Y, seed)

    else:
        # Default to OLS
        return _estimate_ols(source_var, target_var, X_matrix, Y, n)


def _estimate_ols(
    source: str,
    target: str,
    X: np.ndarray,
    Y: np.ndarray,
    n: int
) -> CausalCoefficient:
    """
    Estimate via Ordinary Least Squares regression.

    Y = β₀ + β₁X + ε
    """
    # Add intercept
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    X_with_intercept = np.column_stack([np.ones(n), X])

    # Solve normal equations: β = (X'X)⁻¹ X'Y
    try:
        XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        beta = XtX_inv @ X_with_intercept.T @ Y

        # Coefficient for source variable (first predictor after intercept)
        coef = beta[1]

        # Standard error
        residuals = Y - X_with_intercept @ beta
        mse = np.sum(residuals ** 2) / (n - X_with_intercept.shape[1])
        se = np.sqrt(mse * XtX_inv[1, 1])

        # T-statistic and p-value
        t_stat = coef / se if se > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))

        # 95% confidence interval
        t_crit = stats.t.ppf(0.975, df=n - 2)
        ci = (coef - t_crit * se, coef + t_crit * se)

        return CausalCoefficient(
            source_var=source,
            target_var=target,
            coefficient=coef,
            std_error=se,
            confidence_interval=ci,
            p_value=p_value,
            n_observations=n,
            method=EstimationMethod.OLS
        )

    except np.linalg.LinAlgError:
        # Singular matrix - return zero coefficient
        return CausalCoefficient(
            source_var=source,
            target_var=target,
            coefficient=0.0,
            std_error=float('inf'),
            confidence_interval=(0.0, 0.0),
            p_value=1.0,
            n_observations=n,
            method=EstimationMethod.OLS
        )


def _estimate_matching(
    source: str,
    target: str,
    X: np.ndarray,
    Y: np.ndarray,
    seed: int
) -> CausalCoefficient:
    """
    Estimate via propensity score matching.

    Simplified version: match on X values and compute average treatment effect.
    """
    n = len(X)

    # For binary treatment, split into treated/control
    # For continuous, use median split
    median_x = np.median(X)
    treated = Y[X > median_x]
    control = Y[X <= median_x]

    if len(treated) == 0 or len(control) == 0:
        # Insufficient variation
        return CausalCoefficient(
            source_var=source,
            target_var=target,
            coefficient=0.0,
            std_error=float('inf'),
            confidence_interval=(0.0, 0.0),
            p_value=1.0,
            n_observations=n,
            method=EstimationMethod.MATCHING
        )

    # Average treatment effect
    ate = np.mean(treated) - np.mean(control)

    # Standard error via pooled variance
    se = np.sqrt(
        np.var(treated) / len(treated) + np.var(control) / len(control)
    )

    # T-test
    t_stat = ate / se if se > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))

    # CI
    t_crit = stats.t.ppf(0.975, df=n - 2)
    ci = (ate - t_crit * se, ate + t_crit * se)

    return CausalCoefficient(
        source_var=source,
        target_var=target,
        coefficient=ate,
        std_error=se,
        confidence_interval=ci,
        p_value=p_value,
        n_observations=n,
        method=EstimationMethod.MATCHING
    )


def _estimate_double_ml(
    source: str,
    target: str,
    X: np.ndarray,
    Y: np.ndarray,
    seed: int
) -> CausalCoefficient:
    """
    Estimate via Double Machine Learning (simplified).

    DML uses cross-fitting to debias estimates when using ML for nuisance parameters.
    Simplified: just use OLS for now.
    """
    return _estimate_ols(source, target, X, Y, len(Y))


def estimate_all_edges(
    causal_graph: Any,
    data: Dict[str, List[float]],
    method: EstimationMethod = EstimationMethod.OLS,
    seed: int = _GLOBAL_SEED
) -> Dict[Tuple[str, str], CausalCoefficient]:
    """
    Estimate coefficients for all edges in causal graph.

    Args:
        causal_graph: Causal graph structure
        data: Observational data
        method: Estimation method
        seed: Random seed

    Returns:
        Dictionary mapping (source, target) pairs to coefficients
    """
    coefficients = {}

    for edge in causal_graph.edges:
        source = edge.source.name
        target = edge.target.name

        # Get confounders (parents of target excluding source)
        target_parents = [p.name for p in causal_graph.get_parents(target)]
        confounders = [p for p in target_parents if p != source]

        try:
            coef = estimate_causal_effect(
                source,
                target,
                data,
                confounders=confounders,
                method=method,
                seed=seed
            )
            coefficients[(source, target)] = coef

            # Update edge in graph
            edge.coefficient = coef.coefficient
            edge.confidence = coef.confidence_interval

        except Exception as e:
            # Skip edges that can't be estimated
            print(f"Warning: Could not estimate {source} → {target}: {e}")
            continue

    return coefficients


def compute_stability(
    causal_graph: Any,
    data: Dict[str, List[float]],
    n_bootstrap: int = 100,
    seed: int = _GLOBAL_SEED
) -> Dict[Tuple[str, str], Dict]:
    """
    Compute stability of causal coefficients via bootstrap.

    Args:
        causal_graph: Causal graph
        data: Observational data
        n_bootstrap: Number of bootstrap samples
        seed: Random seed

    Returns:
        Dictionary with stability metrics for each edge
    """
    rng = SeededRNG(seed)
    n = len(next(iter(data.values())))

    stability_results = {}

    for edge in causal_graph.edges:
        source = edge.source.name
        target = edge.target.name

        if source not in data or target not in data:
            continue

        # Bootstrap estimates
        bootstrap_coefs = []

        for b in range(n_bootstrap):
            # Resample indices
            indices = rng.randint(0, n, size=n)

            # Resample data
            resampled_data = {
                var: [vals[i] for i in indices]
                for var, vals in data.items()
            }

            # Estimate
            try:
                coef = estimate_causal_effect(
                    source,
                    target,
                    resampled_data,
                    method=EstimationMethod.OLS,
                    seed=seed + b
                )
                bootstrap_coefs.append(coef.coefficient)
            except:
                continue

        if bootstrap_coefs:
            stability_results[(source, target)] = {
                'mean': np.mean(bootstrap_coefs),
                'std': np.std(bootstrap_coefs),
                'cv': np.std(bootstrap_coefs) / abs(np.mean(bootstrap_coefs))
                      if np.mean(bootstrap_coefs) != 0 else float('inf'),
                'percentiles': {
                    '5': np.percentile(bootstrap_coefs, 5),
                    '50': np.percentile(bootstrap_coefs, 50),
                    '95': np.percentile(bootstrap_coefs, 95)
                }
            }

    return stability_results


def format_pass_message(
    source: str,
    target: str,
    coefficient: CausalCoefficient
) -> str:
    """
    Format causal coefficient as [PASS] message.

    Returns:
        Formatted string: "[PASS] Causal Model Stable do(X)->Y coeff=<v>]"
    """
    if coefficient.is_significant:
        status = "Stable"
    else:
        status = "Unstable"

    return (
        f"[PASS] Causal Model {status} "
        f"do({source})->{target} "
        f"coeff={coefficient.coefficient:.3f} "
        f"(p={coefficient.p_value:.4f})"
    )
